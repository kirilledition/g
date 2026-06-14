"""Core callback lifecycle and bounded-queue runtime used by REGENIE callbacks."""

from __future__ import annotations

import abc
import contextlib
import queue
import threading
import time
import typing
from dataclasses import dataclass

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
HostGenotypeBuffer = shared.HostGenotypeBuffer
PreprocessedDosageChunkWorkItem = shared.PreprocessedDosageChunkWorkItem
PreprocessedVariantMajorDosageChunkWorkItem = shared.PreprocessedVariantMajorDosageChunkWorkItem
PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem = (
    shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
)
Regenie2ResultWriteWorkItem = shared.Regenie2ResultWriteWorkItem
Regenie2MultiResultWriteWorkItem = shared.Regenie2MultiResultWriteWorkItem
NativeBgenWorkerShutdownError = shared.NativeBgenWorkerShutdownError
record_stage_duration_with_optional_chunk = transfers.record_stage_duration_with_optional_chunk
write_regenie2_native_chunk_with_optional_timing = writers.write_regenie2_native_chunk_with_optional_timing
record_binary_chunk_diagnostics_from_count = diagnostics.record_binary_chunk_diagnostics_from_count
binary_chunk_diagnostics_to_mapping = regenie2_binary.binary_chunk_diagnostics_to_mapping
get_metadata_chromosome = shared.get_metadata_chromosome


@dataclass
class BinaryCorrectionSummary:
    """Aggregate binary correction counters accumulated across chunks."""

    score_only_count: int
    score_test_candidate_count: int
    firth_attempted_count: int
    firth_success_count: int
    firth_failed_count: int
    firth_numerical_failure_count: int
    firth_max_iteration_failure_count: int
    firth_invalid_statistic_failure_count: int
    firth_step_halving_failure_count: int
    pseudo_firth_attempt_count: int
    pseudo_firth_success_count: int
    nr_zero_start_attempt_count: int
    nr_zero_start_success_count: int
    nr_warm_start_attempt_count: int
    nr_warm_start_success_count: int
    sparse_correction_count: int
    dense_correction_count: int
    null_model_failure_count: int


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
        result_in_flight_limit: int | None,
        dosage_buffer_limit: int | None,
        stage_timing_recorder: timing.StageTimingRecorder | None,
        telemetry_session: telemetry.TelemetrySession | None,
        output_statistic_dtype: types.FloatingPointDtype,
    ) -> None:
        """Initialize shared native callback state."""
        if staging_depth <= 0:
            message = "staging_depth must be positive."
            raise ValueError(message)
        if result_in_flight_limit is not None and result_in_flight_limit <= 0:
            message = "result_in_flight_limit must be positive when provided."
            raise ValueError(message)
        if dosage_buffer_limit is not None and dosage_buffer_limit <= 0:
            message = "dosage_buffer_limit must be positive when provided."
            raise ValueError(message)
        self.processed_chunk_count = 0
        self.stage_timing_recorder = stage_timing_recorder
        self.telemetry_session = telemetry_session
        self.output_statistic_dtype = output_statistic_dtype
        self.current_progress_chromosome: str | None = None
        self.dosage_queue_depth = staging_depth
        self.result_queue_depth = staging_depth
        self.result_in_flight_limit = result_in_flight_limit or self.result_queue_depth + 1
        self.dosage_buffer_limit = dosage_buffer_limit or self.dosage_queue_depth + 1
        self.result_in_flight_slot_count = 0
        self.result_in_flight_slot_lock = threading.Lock()
        self.dosage_queue: queue.Queue[
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
            | None
        ] = queue.Queue(maxsize=self.dosage_queue_depth)
        self.result_queue: queue.Queue[Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem | None] = (
            queue.Queue(maxsize=self.result_queue_depth)
        )
        self.result_in_flight_slots = threading.BoundedSemaphore(self.result_in_flight_limit)
        self.free_dosage_buffers: queue.Queue[HostGenotypeBuffer] = queue.Queue(maxsize=self.dosage_buffer_limit)
        self.dosage_buffer_count = 0
        self.dosage_buffer_identifiers: set[int] = set()
        self.worker_error: BaseException | None = None
        self.result_worker_error: BaseException | None = None
        self.binary_correction_summary = BinaryCorrectionSummary(
            score_only_count=0,
            score_test_candidate_count=0,
            firth_attempted_count=0,
            firth_success_count=0,
            firth_failed_count=0,
            firth_numerical_failure_count=0,
            firth_max_iteration_failure_count=0,
            firth_invalid_statistic_failure_count=0,
            firth_step_halving_failure_count=0,
            pseudo_firth_attempt_count=0,
            pseudo_firth_success_count=0,
            nr_zero_start_attempt_count=0,
            nr_zero_start_success_count=0,
            nr_warm_start_attempt_count=0,
            nr_warm_start_success_count=0,
            sparse_correction_count=0,
            dense_correction_count=0,
            null_model_failure_count=0,
        )
        self.binary_correction_summary_chunk_count = 0
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
        self.worker_threads_started = False

    def start(self) -> None:
        """Start asynchronous callback workers after owner setup is complete."""
        with self.worker_start_lock:
            if self.worker_threads_started:
                return
            self.result_worker_thread.start()
            self.worker_thread.start()
            self.worker_threads_started = True

    def worker_threads_have_started(self) -> bool:
        """Return whether callback worker threads have been started."""
        return self.worker_threads_started

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
        blocked_seconds: float,
    ) -> None:
        """Record aggregate queue depth and wait metadata."""
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=queue_name,
            operation_name=operation_name,
            queue_depth=observed_queue.qsize(),
            queue_capacity=observed_queue.maxsize,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
        )

    def record_queue_stage_duration(
        self,
        *,
        queue_name: str,
        operation_name: str,
        stage_name: str,
        observed_queue: queue.Queue[typing.Any],
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record a queue stage duration plus aggregate pressure metadata."""
        elapsed_seconds = time.perf_counter() - start_time
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_stage_duration(stage_name, elapsed_seconds)
        blocked_seconds = elapsed_seconds if blocked else 0.0
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=queue_name,
            operation_name=operation_name,
            queue_depth=observed_queue.qsize(),
            queue_capacity=observed_queue.maxsize,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
        )

    def record_bounded_resource_operation(
        self,
        *,
        resource_name: str,
        operation_name: str,
        current_depth: int,
        capacity: int,
        elapsed_seconds: float,
        blocked_seconds: float,
    ) -> None:
        """Record aggregate bounded-resource occupancy metadata."""
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            queue_depth=current_depth,
            queue_capacity=capacity,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
        )

    def record_bounded_resource_stage_duration(
        self,
        *,
        resource_name: str,
        operation_name: str,
        current_depth: int,
        capacity: int,
        stage_name: str,
        start_time: float,
        blocked: bool,
    ) -> None:
        """Record a bounded-resource stage duration plus pressure metadata."""
        elapsed_seconds = time.perf_counter() - start_time
        if self.stage_timing_recorder is None:
            return
        self.stage_timing_recorder.add_stage_duration(stage_name, elapsed_seconds)
        blocked_seconds = elapsed_seconds if blocked else 0.0
        self.stage_timing_recorder.add_queue_backpressure_observation(
            queue_name=resource_name,
            operation_name=operation_name,
            queue_depth=current_depth,
            queue_capacity=capacity,
            elapsed_seconds=elapsed_seconds,
            blocked_seconds=blocked_seconds,
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
                    stage_name="callback_queue_consumer_wait",
                    observed_queue=self.dosage_queue,
                    start_time=get_start_time,
                    blocked=True,
                )
                python_callback_start_time = time.perf_counter()
                try:
                    self.process_dosage_work_item(work_item)
                finally:
                    self.record_chunk_stage_duration(work_item.metadata, "python_callback", python_callback_start_time)
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
            | PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem
        ),
    ) -> None:
        """Run one preprocessed dosage work item."""
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
        self.processed_chunk_count += 1
        self.record_progress(work_item.metadata)

    def record_progress(self, metadata: typing.Any) -> None:
        """Record throttled progress after one chunk is processed."""
        if self.telemetry_session is None:
            return
        chromosome = get_metadata_chromosome(metadata)
        if chromosome != self.current_progress_chromosome:
            if self.current_progress_chromosome is not None:
                self.telemetry_session.log_event(
                    "chromosome_completed",
                    level="info",
                    chromosome=self.current_progress_chromosome,
                    processed_chunk_count=self.processed_chunk_count - 1,
                )
            self.current_progress_chromosome = chromosome
            self.telemetry_session.log_event(
                "chromosome_started",
                level="info",
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

    def record_binary_null_model_failure_count(self, failure_count: int) -> None:
        """Accumulate binary null-model failures for run-level telemetry."""
        self.binary_correction_summary.null_model_failure_count += failure_count

    def record_binary_correction_diagnostics(
        self,
        binary_chunk_diagnostics: regenie2_binary.BinaryChunkDiagnostics | None,
    ) -> None:
        """Accumulate binary correction diagnostics for run-level telemetry."""
        if binary_chunk_diagnostics is None:
            return
        if self.telemetry_session is None:
            return
        diagnostics_mapping = binary_chunk_diagnostics_to_mapping(binary_chunk_diagnostics)
        self.binary_correction_summary_chunk_count += 1
        self.binary_correction_summary.score_only_count += int(diagnostics_mapping["score_only_count"])
        self.binary_correction_summary.score_test_candidate_count += int(
            diagnostics_mapping["score_test_candidate_count"]
        )
        self.binary_correction_summary.firth_attempted_count += int(diagnostics_mapping["firth_candidate_count"])
        self.binary_correction_summary.firth_success_count += int(diagnostics_mapping["firth_converged_count"])
        self.binary_correction_summary.firth_failed_count += int(diagnostics_mapping["firth_failed_count"])
        self.binary_correction_summary.firth_numerical_failure_count += int(
            diagnostics_mapping["firth_numerical_failure_count"]
        )
        self.binary_correction_summary.firth_max_iteration_failure_count += int(
            diagnostics_mapping["firth_max_iteration_failure_count"]
        )
        self.binary_correction_summary.firth_invalid_statistic_failure_count += int(
            diagnostics_mapping["firth_invalid_statistic_failure_count"]
        )
        self.binary_correction_summary.firth_step_halving_failure_count += int(
            diagnostics_mapping["firth_step_halving_failure_count"]
        )
        self.binary_correction_summary.pseudo_firth_attempt_count += int(
            diagnostics_mapping["pseudo_firth_attempt_count"]
        )
        self.binary_correction_summary.pseudo_firth_success_count += int(
            diagnostics_mapping["pseudo_firth_success_count"]
        )
        self.binary_correction_summary.nr_zero_start_attempt_count += int(
            diagnostics_mapping["nr_zero_start_attempt_count"]
        )
        self.binary_correction_summary.nr_zero_start_success_count += int(
            diagnostics_mapping["nr_zero_start_success_count"]
        )
        self.binary_correction_summary.nr_warm_start_attempt_count += int(
            diagnostics_mapping["nr_warm_start_attempt_count"]
        )
        self.binary_correction_summary.nr_warm_start_success_count += int(
            diagnostics_mapping["nr_warm_start_success_count"]
        )
        self.binary_correction_summary.sparse_correction_count += int(diagnostics_mapping["sparse_correction_count"])
        self.binary_correction_summary.dense_correction_count += int(diagnostics_mapping["dense_correction_count"])

    def emit_binary_correction_summary(self) -> None:
        """Emit aggregate binary correction diagnostics when a binary run produced them."""
        if self.telemetry_session is None:
            return
        if (
            self.binary_correction_summary_chunk_count == 0
            and self.binary_correction_summary.null_model_failure_count == 0
        ):
            return
        self.telemetry_session.log_event(
            "binary_correction_summary",
            level="info",
            chunk_count=self.binary_correction_summary_chunk_count,
            score_only_count=self.binary_correction_summary.score_only_count,
            score_test_candidate_count=self.binary_correction_summary.score_test_candidate_count,
            firth_attempted_count=self.binary_correction_summary.firth_attempted_count,
            firth_success_count=self.binary_correction_summary.firth_success_count,
            firth_failed_count=self.binary_correction_summary.firth_failed_count,
            firth_numerical_failure_count=self.binary_correction_summary.firth_numerical_failure_count,
            firth_max_iteration_failure_count=self.binary_correction_summary.firth_max_iteration_failure_count,
            firth_invalid_statistic_failure_count=(
                self.binary_correction_summary.firth_invalid_statistic_failure_count
            ),
            firth_step_halving_failure_count=self.binary_correction_summary.firth_step_halving_failure_count,
            pseudo_firth_attempt_count=self.binary_correction_summary.pseudo_firth_attempt_count,
            pseudo_firth_success_count=self.binary_correction_summary.pseudo_firth_success_count,
            nr_zero_start_attempt_count=self.binary_correction_summary.nr_zero_start_attempt_count,
            nr_zero_start_success_count=self.binary_correction_summary.nr_zero_start_success_count,
            nr_warm_start_attempt_count=self.binary_correction_summary.nr_warm_start_attempt_count,
            nr_warm_start_success_count=self.binary_correction_summary.nr_warm_start_success_count,
            sparse_correction_count=self.binary_correction_summary.sparse_correction_count,
            dense_correction_count=self.binary_correction_summary.dense_correction_count,
            null_model_failure_count=self.binary_correction_summary.null_model_failure_count,
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
                    return
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="consumer_wait",
                    stage_name="result_queue_consumer_wait",
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
                return
            self.process_result_write_item(work_item)

    def process_result_write_item(
        self,
        work_item: Regenie2ResultWriteWorkItem | Regenie2MultiResultWriteWorkItem,
    ) -> None:
        """Materialize and write one computed result work item."""
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
                output_statistic_dtype=self.output_statistic_dtype,
            )
            record_binary_chunk_diagnostics_from_count(
                stage_timing_recorder=self.stage_timing_recorder,
                diagnostics=work_item.binary_chunk_diagnostics,
            )
            self.record_binary_correction_diagnostics(work_item.binary_chunk_diagnostics)
        finally:
            self.release_result_work_item_buffer(work_item)

    def put_dosage_work_item(
        self,
        work_item: (
            PreprocessedDosageChunkWorkItem
            | PreprocessedVariantMajorDosageChunkWorkItem
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
                    self.dosage_queue.put(work_item, timeout=0.1)
                    return
                except queue.Full:
                    continue
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.dosage_queue.put(work_item, timeout=0.1)
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="put",
                    stage_name="callback_queue_put",
                    observed_queue=self.dosage_queue,
                    start_time=put_start_time,
                    blocked=False,
                )
                return
            except queue.Full:
                self.record_queue_stage_duration(
                    queue_name="dosage_queue",
                    operation_name="producer_blocking",
                    stage_name="callback_queue_producer_blocking",
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
                    self.result_queue.put(work_item, timeout=0.1)
                    return
                except queue.Full:
                    continue
        while True:
            self.raise_worker_error_if_present()
            put_start_time = time.perf_counter()
            try:
                self.result_queue.put(work_item, timeout=0.1)
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="put",
                    stage_name="result_queue_put",
                    observed_queue=self.result_queue,
                    start_time=put_start_time,
                    blocked=False,
                )
                return
            except queue.Full:
                self.record_queue_stage_duration(
                    queue_name="result_queue",
                    operation_name="producer_blocking",
                    stage_name="result_queue_producer_blocking",
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
                if self.result_in_flight_slots.acquire(timeout=0.1):
                    with self.result_in_flight_slot_lock:
                        self.result_in_flight_slot_count += 1
                    return
        while True:
            self.raise_worker_error_if_present()
            acquire_start_time = time.perf_counter()
            if self.result_in_flight_slots.acquire(timeout=0.1):
                with self.result_in_flight_slot_lock:
                    self.result_in_flight_slot_count += 1
                    current_depth = self.result_in_flight_slot_count
                self.record_bounded_resource_stage_duration(
                    resource_name="result_in_flight_slots",
                    operation_name="acquire",
                    current_depth=current_depth,
                    capacity=self.result_in_flight_limit,
                    stage_name="result_in_flight_slot_acquire",
                    start_time=acquire_start_time,
                    blocked=False,
                )
                return
            with self.result_in_flight_slot_lock:
                current_depth = self.result_in_flight_slot_count
            self.record_bounded_resource_stage_duration(
                resource_name="result_in_flight_slots",
                operation_name="producer_blocking",
                current_depth=current_depth,
                capacity=self.result_in_flight_limit,
                stage_name="result_in_flight_producer_blocking",
                start_time=acquire_start_time,
                blocked=True,
            )

    def release_result_in_flight_slot(self) -> None:
        """Release capacity for one completed chunk of GPU result work."""
        self.result_in_flight_slots.release()
        with self.result_in_flight_slot_lock:
            if self.result_in_flight_slot_count > 0:
                self.result_in_flight_slot_count -= 1
            current_depth = self.result_in_flight_slot_count
        self.record_bounded_resource_operation(
            resource_name="result_in_flight_slots",
            operation_name="release",
            current_depth=current_depth,
            capacity=self.result_in_flight_limit,
            elapsed_seconds=0.0,
            blocked_seconds=0.0,
        )

    def finish(self) -> None:
        """Wait until all queued JAX work has been written."""
        self.stop_dosage_worker(timeout_seconds=None)
        self.join_dosage_worker(timeout_seconds=GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS)
        self.stop_result_worker(timeout_seconds=None)
        self.join_result_worker(timeout_seconds=GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS)
        self.raise_worker_error_if_present()
        if self.telemetry_session is not None and self.current_progress_chromosome is not None:
            self.telemetry_session.log_event(
                "chromosome_completed",
                level="info",
                chromosome=self.current_progress_chromosome,
                processed_chunk_count=self.processed_chunk_count,
            )
            self.current_progress_chromosome = None
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
        if not self.worker_threads_have_started():
            return
        if self.worker_error is not None:
            return
        if not self.worker_thread.is_alive():
            return
        stop_deadline = time.monotonic() + effective_timeout_seconds
        while time.monotonic() < stop_deadline:
            if self.worker_error is not None:
                return
            if not self.worker_thread.is_alive():
                return
            current_timeout_seconds = max(0.0, min(0.1, stop_deadline - time.monotonic()))
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
        if not self.worker_threads_have_started():
            return
        if self.result_worker_error is not None:
            return
        if not self.result_worker_thread.is_alive():
            return
        stop_deadline = time.monotonic() + effective_timeout_seconds
        while time.monotonic() < stop_deadline:
            if self.result_worker_error is not None:
                return
            if not self.result_worker_thread.is_alive():
                return
            current_timeout_seconds = max(0.0, min(0.1, stop_deadline - time.monotonic()))
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
    def _dosage_buffer_shape_is_compatible(
        buffered_shape: tuple[int, ...],
        expected_shape: tuple[int, ...],
    ) -> bool:
        """Return whether one buffer shape can satisfy another request by slicing."""
        if len(buffered_shape) != len(expected_shape):
            return False
        return all(buffered_dim >= expected_dim for buffered_dim, expected_dim in zip(buffered_shape, expected_shape))

    @classmethod
    def _acquire_reused_dosage_buffer(
        cls,
        dosage_buffer: HostGenotypeBuffer,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> HostGenotypeBuffer | None:
        """Return a reused buffer if dtype/shape constraints are met, else None."""
        if dosage_buffer.dtype != dtype:
            return None
        if dosage_buffer.shape == expected_shape:
            return dosage_buffer
        if not cls._dosage_buffer_shape_is_compatible(dosage_buffer.shape, expected_shape):
            return None
        slices = tuple(slice(0, dimension_size) for dimension_size in expected_shape)
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
                        blocked_seconds=0.0,
                    )
                    return reused_dosage_buffer
                self.discard_dosage_buffer_slot(dosage_buffer)
                if self.dosage_buffer_count < self.dosage_buffer_limit:
                    return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)
                continue
            if self.dosage_buffer_count < self.dosage_buffer_limit:
                return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)
            with contextlib.suppress(queue.Empty):
                if self.stage_timing_recorder is None:
                    dosage_buffer = self.free_dosage_buffers.get(timeout=0.1)
                else:
                    buffer_wait_start_time = time.perf_counter()
                    dosage_buffer = self.free_dosage_buffers.get(timeout=0.1)
                    self.record_queue_stage_duration(
                        queue_name="dosage_buffer_pool",
                        operation_name="consumer_wait",
                        stage_name="dosage_buffer_pool_consumer_wait",
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
                        blocked_seconds=0.0,
                    )
                    return reused_dosage_buffer
                self.discard_dosage_buffer_slot(dosage_buffer)
                if self.dosage_buffer_count < self.dosage_buffer_limit:
                    return self.allocate_dosage_buffer_with_shape(expected_shape, dtype)

    def release_dosage_buffer(self, dosage_buffer: HostGenotypeBuffer) -> None:
        """Return a processed host dosage buffer to the reusable pool."""
        dosage_buffer_owner = self._dosage_buffer_owner(dosage_buffer)
        if id(dosage_buffer_owner) not in self.dosage_buffer_identifiers:
            return
        try:
            self.free_dosage_buffers.put_nowait(dosage_buffer_owner)
            self.record_queue_operation(
                queue_name="dosage_buffer_pool",
                operation_name="return",
                observed_queue=self.free_dosage_buffers,
                elapsed_seconds=0.0,
                blocked_seconds=0.0,
            )
        except queue.Full:
            self.record_queue_operation(
                queue_name="dosage_buffer_pool",
                operation_name="return_full",
                observed_queue=self.free_dosage_buffers,
                elapsed_seconds=0.0,
                blocked_seconds=0.0,
            )
            self.discard_dosage_buffer_slot(dosage_buffer)

    def allocate_dosage_buffer_with_shape(
        self,
        expected_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
    ) -> HostGenotypeBuffer:
        """Allocate and register one host genotype buffer slot."""
        dosage_buffer = typing.cast("HostGenotypeBuffer", np.empty(expected_shape, dtype=dtype, order="C"))
        self.dosage_buffer_count += 1
        self.dosage_buffer_identifiers.add(id(dosage_buffer))
        self.record_queue_operation(
            queue_name="dosage_buffer_pool",
            operation_name="allocate",
            observed_queue=self.free_dosage_buffers,
            elapsed_seconds=0.0,
            blocked_seconds=0.0,
        )
        return dosage_buffer

    def discard_dosage_buffer_slot(self, dosage_buffer: HostGenotypeBuffer) -> None:
        """Remove one discarded host genotype buffer slot from pool accounting."""
        dosage_buffer_identifier = id(dosage_buffer)
        if dosage_buffer_identifier not in self.dosage_buffer_identifiers:
            return
        self.dosage_buffer_identifiers.remove(dosage_buffer_identifier)
        if self.dosage_buffer_count > 0:
            self.dosage_buffer_count -= 1
        self.record_queue_operation(
            queue_name="dosage_buffer_pool",
            operation_name="discard",
            observed_queue=self.free_dosage_buffers,
            elapsed_seconds=0.0,
            blocked_seconds=0.0,
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
            if id(dosage_buffer_owner) in self.dosage_buffer_identifiers:
                return dosage_buffer_owner
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


__all__ = [
    "DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_DOSAGE_WORKER_JOIN_TIMEOUT_SECONDS",
    "GRACEFUL_RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "RESULT_WORKER_JOIN_TIMEOUT_SECONDS",
    "WORKER_ABORT_STOP_TIMEOUT_SECONDS",
    "NativeBgenCallbackRunner",
    "require_current_chromosome_state",
]
