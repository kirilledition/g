"""Thin Python callback glue over native-owned callback runtime resources.

Worker loops, queue capacity, slots, dosage buffers, and lifecycle policy are
owned by ``NativeCallbackRuntimeResources``. This module only stores local
callback state and implements JAX/write side effects.
"""

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


class ResultWriteItemKind(enum.StrEnum):
    """Native result-write work-item kind values."""

    SINGLE_RESULT = "single_result"
    MULTI_RESULT = "multi_result"


class NativeBgenCallbackRunner(abc.ABC):
    """Python-side callback hooks over native-owned callback runtime resources."""

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
            dosage_worker_target=self._dosage_worker_entry,
            result_worker_target=self._result_worker_entry,
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
    def native_callback_batch_size(self) -> int:
        """Return the native callback batch size."""
        return self.callback_runtime_resources.native_callback_batch_size

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
        self, metadata: typing.Any, stage_name: str, elapsed_seconds: float
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
        work_item: shared.PreprocessedDosageChunkWorkItem
        | shared.PreprocessedVariantMajorDosageChunkWorkItem
        | shared.PreprocessedVariantMajorDosageChunkBatchWorkItem
        | shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem,
        stage_name: str,
        elapsed_seconds: float,
    ) -> None:
        """Record a stage duration across one queued work item."""
        attribution = self.callback_runtime_resources.plan_dosage_work_item_stage_duration_attribution_for_object(
            work_item, elapsed_seconds
        )
        chunk_metadata_items = attribution.metadata_items
        stage_duration_plan = attribution.stage_duration_plan
        if stage_duration_plan.chunk_count != len(chunk_metadata_items):
            message = "Native dosage work stage duration plan disagrees with the work item chunk count."
            raise RuntimeError(message)
        for chunk_metadata in chunk_metadata_items:
            self.record_chunk_stage_elapsed_duration(chunk_metadata, stage_name, stage_duration_plan.duration_per_chunk)

    def get_stage_duration_recorder(self) -> collections.abc.Callable[[str, float], None] | None:
        """Return an optional nested stage recorder for lower-level compute helpers."""
        if self.stage_timing_recorder is None:
            return None
        return self.record_stage_duration

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

    def _dosage_worker_entry(self) -> None:
        """Thread entry: run the native-owned dosage worker loop."""
        try:
            self.callback_runtime_resources.run_dosage_worker_loop(self)
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

    def _result_worker_entry(self) -> None:
        """Thread entry: run the native-owned result-write worker loop."""
        try:
            self.callback_runtime_resources.run_result_worker_loop(self)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def process_dosage_work_item_with_dispatch_outcome(
        self,
        work_item: shared.PreprocessedDosageChunkWorkItem
        | shared.PreprocessedVariantMajorDosageChunkWorkItem
        | shared.PreprocessedVariantMajorDosageChunkBatchWorkItem
        | shared.PreprocessedVariantMajorPacked8ProbabilityPairChunkWorkItem,
        dispatch_outcome: typing.Any,
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
        self, work_item: shared.PreprocessedVariantMajorDosageChunkBatchWorkItem
    ) -> None:
        """Run one native-planned variant-major dosage batch work item."""
        for chunk_work_item in work_item.work_items:
            self.compute_preprocessed_variant_major_chunk(
                variant_metadata=chunk_work_item.metadata,
                genotype_matrix_by_variant=chunk_work_item.genotype_matrix_by_variant,
                chunk_stats=chunk_work_item.chunk_stats,
            )
            self.record_progress(chunk_work_item.metadata)

    def record_progress(self, metadata: typing.Any) -> None:
        """Record throttled progress after one chunk is processed."""
        self.callback_runtime_resources.record_progress_and_emit_telemetry(metadata, self.telemetry_session)

    def record_binary_null_model_failure_count(self, failure_count: int) -> None:
        """Accumulate binary null-model failures for run-level telemetry."""
        self.callback_runtime_resources.add_binary_null_model_failure_count(failure_count)

    def record_binary_correction_diagnostics(
        self, binary_chunk_diagnostics: regenie2_binary.BinaryChunkDiagnostics | None
    ) -> None:
        """Accumulate binary correction diagnostics for run-level telemetry."""
        should_record_diagnostics = (
            self.callback_runtime_resources.should_record_binary_correction_diagnostics_for_object(
                binary_chunk_diagnostics
            )
        )
        if not should_record_diagnostics:
            return
        if binary_chunk_diagnostics is None:
            message = "Native binary correction diagnostics record plan selected a missing diagnostics payload."
            raise RuntimeError(message)
        self.binary_correction_pending_diagnostics.append(binary_chunk_diagnostics)

    def flush_binary_correction_diagnostics(self) -> None:
        """Materialize pending binary diagnostics and accumulate them into native summary counters."""
        should_flush_pending_diagnostics = (
            self.callback_runtime_resources.should_flush_binary_correction_pending_diagnostics(
                self.binary_correction_pending_diagnostics
            )
        )
        if not should_flush_pending_diagnostics:
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
        self, diagnostics_counts: regenie2_binary.BinaryCorrectionSummaryCounts
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

    def process_result_write_item(self, work_item: shared.Regenie2ResultWriteWorkItem) -> None:
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
            write_start_time = time.perf_counter() if self.stage_timing_recorder is not None else 0.0
            _core.write_regenie2_native_chunk_with_output_dtype(
                writer_session=typing.cast("_core.OutputWriterSession", typing.cast("typing.Any", self).writer_session),
                metadata=work_item.metadata,
                chunk_stats=work_item.chunk_stats,
                output_statistic_dtype=self.output_statistic_dtype.value,
                beta=transfers.cast_statistic_array_for_native_writer(
                    materialized_chunk.beta, self.output_statistic_dtype
                ),
                standard_error=transfers.cast_statistic_array_for_native_writer(
                    materialized_chunk.standard_error, self.output_statistic_dtype
                ),
                chi_squared=transfers.cast_statistic_array_for_native_writer(
                    materialized_chunk.chi_squared, self.output_statistic_dtype
                ),
                log10_p_value=transfers.cast_statistic_array_for_native_writer(
                    materialized_chunk.log10_p_value, self.output_statistic_dtype
                ),
                extra_code=typing.cast("typing.Any", materialized_chunk.extra_code),
            )
            if self.stage_timing_recorder is not None:
                transfers.record_stage_duration_with_optional_chunk(
                    stage_timing_recorder=self.stage_timing_recorder,
                    stage_name="output_write",
                    start_time=write_start_time,
                    chunk_metadata=work_item.metadata,
                )
                transfers.record_stage_duration_with_optional_chunk(
                    stage_timing_recorder=self.stage_timing_recorder,
                    stage_name="single_trait_output_write",
                    start_time=write_start_time,
                    chunk_metadata=work_item.metadata,
                )
            diagnostics.record_binary_chunk_diagnostics_from_count(
                stage_timing_recorder=self.stage_timing_recorder, diagnostics=work_item.binary_chunk_diagnostics
            )
            self.record_binary_correction_diagnostics(work_item.binary_chunk_diagnostics)
        finally:
            self.release_result_work_item_final_resources(
                work_item, host_dosage_buffer_released=host_dosage_buffer_released
            )

    def process_multi_result_write_item(self, multi_work_item: shared.Regenie2MultiResultWriteWorkItem) -> None:
        """Materialize and write one multi-result work item."""
        del multi_work_item
        raise NotImplementedError

    def raise_worker_error_if_present(self) -> None:
        """Raise an asynchronous worker failure on the producer thread."""
        error_raise_plan = self.callback_runtime_resources.plan_worker_error_raise()
        if error_raise_plan is None or not error_raise_plan.should_raise:
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

    def put_result_write_item(self, work_item: QueuedResultWriteWorkItem) -> None:
        """Put a computed result into the bounded materialization/write queue."""
        native_recorder = None if self.stage_timing_recorder is None else self.stage_timing_recorder.native_recorder
        self.callback_runtime_resources.put_result_write_item_handling_errors(work_item, native_recorder)

    def acquire_result_in_flight_slot(self) -> None:
        """Reserve capacity for one chunk of pending GPU result work."""
        native_recorder = None if self.stage_timing_recorder is None else self.stage_timing_recorder.native_recorder
        self.callback_runtime_resources.acquire_result_in_flight_slot_handling_errors(native_recorder)

    def release_result_in_flight_slot(self) -> None:
        """Release capacity for one completed chunk of GPU result work."""
        native_recorder = None if self.stage_timing_recorder is None else self.stage_timing_recorder.native_recorder
        self.callback_runtime_resources.release_result_in_flight_slot_handling_timing(native_recorder)

    def release_dosage_buffer(self, dosage_buffer: shared.HostGenotypeBuffer) -> None:
        """Return a processed host dosage buffer to the reusable pool."""
        native_recorder = None if self.stage_timing_recorder is None else self.stage_timing_recorder.native_recorder
        self.callback_runtime_resources.release_dosage_buffer_handling_timing(dosage_buffer, native_recorder)

    def get_releasable_dosage_buffer(
        self, dosage_buffer: jax.Array | shared.HostGenotypeBuffer
    ) -> shared.HostGenotypeBuffer | None:
        """Return a host dosage buffer reference when it belongs to the reusable pool."""
        return typing.cast(
            "shared.HostGenotypeBuffer | None",
            self.callback_runtime_resources.get_releasable_dosage_buffer_owner(dosage_buffer),
        )

    def release_result_work_item_host_buffer(
        self, work_item: shared.Regenie2ResultWriteWorkItem | shared.Regenie2MultiResultWriteWorkItem
    ) -> bool:
        """Release the host genotype buffer associated with one result."""
        native_recorder = None if self.stage_timing_recorder is None else self.stage_timing_recorder.native_recorder
        return self.callback_runtime_resources.release_result_work_item_pre_write_resources_handling_timing(
            work_item, native_recorder
        )

    def release_result_work_item_final_resources(
        self,
        work_item: shared.Regenie2ResultWriteWorkItem | shared.Regenie2MultiResultWriteWorkItem,
        *,
        host_dosage_buffer_released: bool,
    ) -> None:
        """Release final resources associated with one result work item."""
        native_recorder = None if self.stage_timing_recorder is None else self.stage_timing_recorder.native_recorder
        self.callback_runtime_resources.release_result_work_item_final_resources_handling_timing(
            work_item, host_dosage_buffer_released, native_recorder
        )
