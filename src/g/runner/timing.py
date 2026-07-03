"""Runner-owned timing diagnostics for REGENIE step 2 runs."""

from __future__ import annotations

import pathlib
import time
import typing
from dataclasses import dataclass

from g import _core

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


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
class BinaryChunkDiagnosticsSnapshot:
    """Host-side binary diagnostics counters for one processed chunk.

    Attributes:
        score_only_count: Variants that retained score-test statistics.
        score_test_candidate_count: Variants selected for score-test fallback labels.
        firth_candidate_count: Variants with a nonzero Firth iteration count.
        firth_iteration_min: Minimum Firth iteration count among candidates.
        firth_iteration_median: Median Firth iteration count among candidates.
        firth_iteration_max: Maximum Firth iteration count among candidates.
        firth_converged_count: Variants with successful Firth correction.
        firth_failed_count: Variants labelled as failed candidate tests.
        firth_numerical_failure_count: Firth numerical failures.
        firth_max_iteration_failure_count: Firth iteration-limit failures.
        firth_invalid_statistic_failure_count: Firth invalid-statistic failures.
        firth_step_halving_failure_count: Firth step-halving failures.
        pseudo_firth_attempt_count: Scalar pseudo-Firth attempts.
        pseudo_firth_success_count: Scalar pseudo-Firth successes.
        nr_zero_start_attempt_count: Zero-start Newton-Raphson attempts.
        nr_zero_start_success_count: Zero-start Newton-Raphson successes.
        nr_warm_start_attempt_count: Warm-start Newton-Raphson attempts.
        nr_warm_start_success_count: Warm-start Newton-Raphson successes.
        sparse_correction_count: Sparse carrier-only corrections.
        dense_correction_count: Dense corrections.

    """

    score_only_count: int | float | None
    score_test_candidate_count: int | float | None
    firth_candidate_count: int | float | None
    firth_iteration_min: int | float | None
    firth_iteration_median: int | float | None
    firth_iteration_max: int | float | None
    firth_converged_count: int | float | None
    firth_failed_count: int | float | None
    firth_numerical_failure_count: int | float | None
    firth_max_iteration_failure_count: int | float | None
    firth_invalid_statistic_failure_count: int | float | None
    firth_step_halving_failure_count: int | float | None
    pseudo_firth_attempt_count: int | float | None
    pseudo_firth_success_count: int | float | None
    nr_zero_start_attempt_count: int | float | None
    nr_zero_start_success_count: int | float | None
    nr_warm_start_attempt_count: int | float | None
    nr_warm_start_success_count: int | float | None
    sparse_correction_count: int | float | None
    dense_correction_count: int | float | None


@dataclass(frozen=True)
class NullLogisticDiagnosticsSnapshot:
    """Host-side null logistic diagnostics for one chromosome or trait lane.

    Attributes:
        chromosome: Chromosome label.
        phenotype: Phenotype name for multi-trait diagnostics.
        iteration_count: Null logistic iteration count.
        converged: Whether null logistic fitting converged, encoded as an integer.
        firth_iteration_count: Null Firth fallback iteration count.
        firth_convergence_reason_code: Null Firth convergence reason code.
        correction_method: Binary correction method.

    """

    chromosome: str | None
    phenotype: str | None
    iteration_count: int | None
    converged: int | None
    firth_iteration_count: int | None
    firth_convergence_reason_code: int | None
    correction_method: str | None


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
    binary_chunk_diagnostics: tuple[BinaryChunkDiagnosticsSnapshot, ...]
    null_logistic_diagnostics: tuple[NullLogisticDiagnosticsSnapshot, ...]
    queue_backpressure: tuple[QueueBackpressureSnapshot, ...]
    transfer_metadata: tuple[TransferMetadataSnapshot, ...]


@dataclass(frozen=True)
class FinalTimingOutputContext:
    """Resolved final timing output paths and recorder policy.

    Attributes:
        stage_timing_path: Optional exact stage timing output path.
        profile_summary_path: Optional profile summary output path.
        run_id: Optional telemetry run identifier for profile summaries.
        force_stage_timing_recorder: Whether aggregate timing should be forced.

    """

    stage_timing_path: pathlib.Path | None
    profile_summary_path: pathlib.Path | None
    run_id: str | None
    force_stage_timing_recorder: bool


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


def optional_numeric_diagnostic(
    diagnostics: typing.Mapping[str, int | float],
    key: str,
) -> int | float | None:
    """Return a numeric diagnostic value when present."""
    return diagnostics.get(key)


def optional_integer_diagnostic(
    diagnostics: typing.Mapping[str, int | str],
    key: str,
) -> int | None:
    """Return an integer diagnostic value when present."""
    value = diagnostics.get(key)
    if value is None:
        return None
    return int(value)


def optional_string_diagnostic(
    diagnostics: typing.Mapping[str, int | str],
    key: str,
) -> str | None:
    """Return a string diagnostic value when present."""
    value = diagnostics.get(key)
    if value is None:
        return None
    return str(value)


def binary_chunk_diagnostics_snapshot_from_mapping(
    diagnostics: typing.Mapping[str, int | float],
) -> BinaryChunkDiagnosticsSnapshot:
    """Build a typed binary diagnostic snapshot from JSON-like counters."""
    return BinaryChunkDiagnosticsSnapshot(
        score_only_count=optional_numeric_diagnostic(diagnostics, "score_only_count"),
        score_test_candidate_count=optional_numeric_diagnostic(diagnostics, "score_test_candidate_count"),
        firth_candidate_count=optional_numeric_diagnostic(diagnostics, "firth_candidate_count"),
        firth_iteration_min=optional_numeric_diagnostic(diagnostics, "firth_iteration_min"),
        firth_iteration_median=optional_numeric_diagnostic(diagnostics, "firth_iteration_median"),
        firth_iteration_max=optional_numeric_diagnostic(diagnostics, "firth_iteration_max"),
        firth_converged_count=optional_numeric_diagnostic(diagnostics, "firth_converged_count"),
        firth_failed_count=optional_numeric_diagnostic(diagnostics, "firth_failed_count"),
        firth_numerical_failure_count=optional_numeric_diagnostic(diagnostics, "firth_numerical_failure_count"),
        firth_max_iteration_failure_count=optional_numeric_diagnostic(diagnostics, "firth_max_iteration_failure_count"),
        firth_invalid_statistic_failure_count=optional_numeric_diagnostic(
            diagnostics,
            "firth_invalid_statistic_failure_count",
        ),
        firth_step_halving_failure_count=optional_numeric_diagnostic(
            diagnostics,
            "firth_step_halving_failure_count",
        ),
        pseudo_firth_attempt_count=optional_numeric_diagnostic(diagnostics, "pseudo_firth_attempt_count"),
        pseudo_firth_success_count=optional_numeric_diagnostic(diagnostics, "pseudo_firth_success_count"),
        nr_zero_start_attempt_count=optional_numeric_diagnostic(diagnostics, "nr_zero_start_attempt_count"),
        nr_zero_start_success_count=optional_numeric_diagnostic(diagnostics, "nr_zero_start_success_count"),
        nr_warm_start_attempt_count=optional_numeric_diagnostic(diagnostics, "nr_warm_start_attempt_count"),
        nr_warm_start_success_count=optional_numeric_diagnostic(diagnostics, "nr_warm_start_success_count"),
        sparse_correction_count=optional_numeric_diagnostic(diagnostics, "sparse_correction_count"),
        dense_correction_count=optional_numeric_diagnostic(diagnostics, "dense_correction_count"),
    )


def null_logistic_diagnostics_snapshot_from_mapping(
    diagnostics: typing.Mapping[str, int | str],
) -> NullLogisticDiagnosticsSnapshot:
    """Build a typed null logistic diagnostic snapshot from JSON-like counters."""
    return NullLogisticDiagnosticsSnapshot(
        chromosome=optional_string_diagnostic(diagnostics, "chromosome"),
        phenotype=optional_string_diagnostic(diagnostics, "phenotype"),
        iteration_count=optional_integer_diagnostic(diagnostics, "iteration_count"),
        converged=optional_integer_diagnostic(diagnostics, "converged"),
        firth_iteration_count=optional_integer_diagnostic(diagnostics, "firth_iteration_count"),
        firth_convergence_reason_code=optional_integer_diagnostic(diagnostics, "firth_convergence_reason_code"),
        correction_method=optional_string_diagnostic(diagnostics, "correction_method"),
    )


class StageTimingRecorder:
    """Thread-safe diagnostic wall-time collector for profiling harnesses."""

    def __init__(self, *, exact_stage_timings: bool) -> None:
        """Initialize empty stage timing state."""
        self.native_recorder = _core.NativeStageTimingRecorder(exact_stage_timings)

    @classmethod
    def from_native_recorder(cls, native_recorder: _core.NativeStageTimingRecorder) -> typing.Self:
        """Build a Python adapter around an existing native recorder."""
        recorder = typing.cast("typing.Self", cls.__new__(cls))
        recorder.native_recorder = native_recorder
        return recorder

    @property
    def exact_stage_timings(self) -> bool:
        """Return whether exact synchronized stage timings are requested."""
        return self.native_recorder.exact_stage_timings

    def should_collect_exact_stage_timings(self) -> bool:
        """Return whether timing should force synchronized exact stage measurements."""
        return self.native_recorder.should_collect_exact_stage_timings()

    def add_stage_duration(self, stage_name: str, duration_seconds: float) -> None:
        """Accumulate one measured duration."""
        self.native_recorder.add_stage_duration(stage_name, duration_seconds)

    def add_chunk_stage_duration(
        self,
        *,
        chunk_identity: ChunkTimingIdentity,
        stage_name: str,
        duration_seconds: float,
    ) -> None:
        """Accumulate one measured duration and attach it to a native chunk."""
        self.native_recorder.add_chunk_stage_duration(
            chunk_identity.chunk_identifier,
            chunk_identity.chromosome,
            chunk_identity.variant_start_index,
            chunk_identity.variant_stop_index,
            chunk_identity.variant_count,
            stage_name,
            duration_seconds,
        )

    def set_native_bgen_profile(self, profile_snapshot: dict[str, int]) -> None:
        """Store native BGEN profiling counters."""
        self.native_recorder.set_native_bgen_profile(profile_snapshot)

    def add_binary_chunk_diagnostics(self, diagnostics: dict[str, int | float]) -> None:
        """Store diagnostic counters for one binary chunk."""
        self.native_recorder.add_binary_chunk_diagnostics(diagnostics)

    def add_null_logistic_diagnostics(self, diagnostics: dict[str, int | str]) -> None:
        """Store null logistic fit diagnostics for one chromosome."""
        self.native_recorder.add_null_logistic_diagnostics(diagnostics)

    def add_scalar_null_logistic_diagnostics_from_arrays(
        self,
        *,
        chromosome: str,
        convergence_values: npt.NDArray[np.bool_],
        iteration_count_values: npt.NDArray[np.int64],
        firth_iteration_count_values: npt.NDArray[np.int64],
        firth_convergence_reason_code_values: npt.NDArray[np.int64],
        correction_method: str,
    ) -> None:
        """Store scalar null logistic diagnostics from native array scans."""
        self.native_recorder.add_scalar_null_logistic_diagnostics_from_arrays(
            chromosome,
            convergence_values,
            iteration_count_values,
            firth_iteration_count_values,
            firth_convergence_reason_code_values,
            correction_method,
        )

    def add_multi_null_logistic_diagnostics_from_arrays(
        self,
        *,
        chromosome: str,
        convergence_values: npt.NDArray[np.bool_],
        iteration_count_values: npt.NDArray[np.int64],
        phenotype_names: tuple[str, ...],
        correction_method: str,
    ) -> None:
        """Store multi-trait null logistic diagnostics from native array scans."""
        self.native_recorder.add_multi_null_logistic_diagnostics_from_arrays(
            chromosome,
            convergence_values,
            iteration_count_values,
            phenotype_names,
            correction_method,
        )

    def add_queue_backpressure_observation(
        self,
        *,
        queue_name: str,
        operation_name: str,
        queue_depth: int,
        queue_capacity: int,
        elapsed_seconds: float,
        blocked_seconds: float,
    ) -> None:
        """Store one queue or bounded-resource pressure observation."""
        self.native_recorder.add_queue_backpressure_observation(
            queue_name,
            operation_name,
            queue_depth,
            queue_capacity,
            elapsed_seconds,
            blocked_seconds,
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
        self.native_recorder.add_transfer_metadata(
            transfer_name,
            array_role,
            dtype_name,
            ndim,
            byte_count,
            element_count,
        )

    def add_transfer_metadata_for_shape(
        self,
        *,
        transfer_name: str,
        array_role: str,
        dtype_name: str,
        shape_dimensions: tuple[int, ...],
        item_size: int,
    ) -> None:
        """Store transfer metadata computed from array shape and dtype item size."""
        self.native_recorder.add_transfer_metadata_for_shape(
            transfer_name,
            array_role,
            dtype_name,
            shape_dimensions,
            item_size,
        )

    def snapshot(self) -> StageTimingSnapshot:
        """Return an immutable copy of the current timings."""
        return adapt_stage_timing_snapshot(self.native_recorder.snapshot())

    def write_final_timing_outputs(
        self,
        *,
        stage_timing_path: pathlib.Path | None,
        profile_summary_path: pathlib.Path | None,
        run_id: str | None,
    ) -> dict[str, bool]:
        """Persist all configured final timing outputs through the native recorder."""
        return dict(
            typing.cast(
                "typing.Mapping[str, bool]",
                self.native_recorder.write_final_timing_outputs(
                    None if stage_timing_path is None else str(stage_timing_path),
                    None if profile_summary_path is None else str(profile_summary_path),
                    run_id,
                ),
            )
        )


def adapt_stage_timing_snapshot(native_snapshot: _core.NativeStageTimingSnapshot) -> StageTimingSnapshot:
    """Adapt a native timing snapshot handle to the public Python shape."""
    return StageTimingSnapshot(
        stage_totals_seconds=dict(native_snapshot.stage_totals_seconds),
        stage_counts=dict(native_snapshot.stage_counts),
        chunk_stage_timings=tuple(
            adapt_chunk_stage_timing(native_chunk_stage_timing)
            for native_chunk_stage_timing in native_snapshot.chunk_stage_timings
        ),
        native_bgen_profile=dict(native_snapshot.native_bgen_profile),
        binary_chunk_diagnostics=tuple(
            binary_chunk_diagnostics_snapshot_from_mapping(binary_diagnostic_payload)
            for binary_diagnostic_payload in native_snapshot.binary_chunk_diagnostics
        ),
        null_logistic_diagnostics=tuple(
            null_logistic_diagnostics_snapshot_from_mapping(null_logistic_diagnostic_payload)
            for null_logistic_diagnostic_payload in native_snapshot.null_logistic_diagnostics
        ),
        queue_backpressure=tuple(
            adapt_queue_backpressure(native_queue_backpressure)
            for native_queue_backpressure in native_snapshot.queue_backpressure
        ),
        transfer_metadata=tuple(
            adapt_transfer_metadata(native_transfer_metadata)
            for native_transfer_metadata in native_snapshot.transfer_metadata
        ),
    )


def adapt_stage_timing_snapshot_payload(snapshot_payload: dict[str, object]) -> StageTimingSnapshot:
    """Adapt a native timing snapshot payload to the public Python shape."""
    return StageTimingSnapshot(
        stage_totals_seconds=dict(typing.cast("typing.Mapping[str, float]", snapshot_payload["stage_totals_seconds"])),
        stage_counts=dict(typing.cast("typing.Mapping[str, int]", snapshot_payload["stage_counts"])),
        chunk_stage_timings=tuple(
            adapt_chunk_stage_timing_payload(chunk_stage_timing_payload)
            for chunk_stage_timing_payload in typing.cast(
                "typing.Sequence[dict[str, object]]",
                snapshot_payload["chunk_stage_timings"],
            )
        ),
        native_bgen_profile=dict(typing.cast("typing.Mapping[str, int]", snapshot_payload["native_bgen_profile"])),
        binary_chunk_diagnostics=tuple(
            binary_chunk_diagnostics_snapshot_from_mapping(
                typing.cast("typing.Mapping[str, int | float]", binary_diagnostic_payload)
            )
            for binary_diagnostic_payload in typing.cast(
                "typing.Sequence[dict[str, object]]",
                snapshot_payload["binary_chunk_diagnostics"],
            )
        ),
        null_logistic_diagnostics=tuple(
            null_logistic_diagnostics_snapshot_from_mapping(
                typing.cast("typing.Mapping[str, int | str]", null_logistic_diagnostic_payload)
            )
            for null_logistic_diagnostic_payload in typing.cast(
                "typing.Sequence[dict[str, object]]",
                snapshot_payload["null_logistic_diagnostics"],
            )
        ),
        queue_backpressure=tuple(
            adapt_queue_backpressure_payload(queue_backpressure_payload)
            for queue_backpressure_payload in typing.cast(
                "typing.Sequence[dict[str, object]]",
                snapshot_payload["queue_backpressure"],
            )
        ),
        transfer_metadata=tuple(
            adapt_transfer_metadata_payload(transfer_metadata_payload)
            for transfer_metadata_payload in typing.cast(
                "typing.Sequence[dict[str, object]]",
                snapshot_payload["transfer_metadata"],
            )
        ),
    )


def adapt_chunk_stage_timing(native_timing: _core.NativeChunkStageTimingSnapshot) -> ChunkStageTimingSnapshot:
    """Adapt one native chunk-stage timing handle."""
    return ChunkStageTimingSnapshot(
        chunk_identifier=native_timing.chunk_identifier,
        chromosome=native_timing.chromosome,
        variant_start_index=native_timing.variant_start_index,
        variant_stop_index=native_timing.variant_stop_index,
        variant_count=native_timing.variant_count,
        stage_name=native_timing.stage_name,
        duration_seconds=native_timing.duration_seconds,
    )


def adapt_chunk_stage_timing_payload(chunk_stage_timing_payload: dict[str, object]) -> ChunkStageTimingSnapshot:
    """Adapt one native chunk-stage timing payload."""
    return ChunkStageTimingSnapshot(
        chunk_identifier=typing.cast("int", chunk_stage_timing_payload["chunk_identifier"]),
        chromosome=typing.cast("str", chunk_stage_timing_payload["chromosome"]),
        variant_start_index=typing.cast("int", chunk_stage_timing_payload["variant_start_index"]),
        variant_stop_index=typing.cast("int", chunk_stage_timing_payload["variant_stop_index"]),
        variant_count=typing.cast("int", chunk_stage_timing_payload["variant_count"]),
        stage_name=typing.cast("str", chunk_stage_timing_payload["stage_name"]),
        duration_seconds=typing.cast("float", chunk_stage_timing_payload["duration_seconds"]),
    )


def adapt_queue_backpressure(
    native_queue_backpressure: _core.NativeQueueBackpressureSnapshot,
) -> QueueBackpressureSnapshot:
    """Adapt one native queue/backpressure handle."""
    return QueueBackpressureSnapshot(
        queue_name=native_queue_backpressure.queue_name,
        operation_name=native_queue_backpressure.operation_name,
        observation_count=native_queue_backpressure.observation_count,
        max_depth=native_queue_backpressure.max_depth,
        max_capacity=native_queue_backpressure.max_capacity,
        total_elapsed_seconds=native_queue_backpressure.total_elapsed_seconds,
        total_blocked_seconds=native_queue_backpressure.total_blocked_seconds,
    )


def adapt_queue_backpressure_payload(queue_backpressure_payload: dict[str, object]) -> QueueBackpressureSnapshot:
    """Adapt one native queue/backpressure payload."""
    return QueueBackpressureSnapshot(
        queue_name=typing.cast("str", queue_backpressure_payload["queue_name"]),
        operation_name=typing.cast("str", queue_backpressure_payload["operation_name"]),
        observation_count=typing.cast("int", queue_backpressure_payload["observation_count"]),
        max_depth=typing.cast("int", queue_backpressure_payload["max_depth"]),
        max_capacity=typing.cast("int", queue_backpressure_payload["max_capacity"]),
        total_elapsed_seconds=typing.cast("float", queue_backpressure_payload["total_elapsed_seconds"]),
        total_blocked_seconds=typing.cast("float", queue_backpressure_payload["total_blocked_seconds"]),
    )


def adapt_transfer_metadata(native_transfer_metadata: _core.NativeTransferMetadataSnapshot) -> TransferMetadataSnapshot:
    """Adapt one native transfer metadata handle."""
    return TransferMetadataSnapshot(
        transfer_name=native_transfer_metadata.transfer_name,
        array_role=native_transfer_metadata.array_role,
        dtype_name=native_transfer_metadata.dtype_name,
        ndim=native_transfer_metadata.ndim,
        observation_count=native_transfer_metadata.observation_count,
        total_bytes=native_transfer_metadata.total_bytes,
        max_bytes=native_transfer_metadata.max_bytes,
        total_elements=native_transfer_metadata.total_elements,
    )


def adapt_transfer_metadata_payload(transfer_metadata_payload: dict[str, object]) -> TransferMetadataSnapshot:
    """Adapt one native transfer metadata payload."""
    return TransferMetadataSnapshot(
        transfer_name=typing.cast("str", transfer_metadata_payload["transfer_name"]),
        array_role=typing.cast("str", transfer_metadata_payload["array_role"]),
        dtype_name=typing.cast("str", transfer_metadata_payload["dtype_name"]),
        ndim=typing.cast("int", transfer_metadata_payload["ndim"]),
        observation_count=typing.cast("int", transfer_metadata_payload["observation_count"]),
        total_bytes=typing.cast("int", transfer_metadata_payload["total_bytes"]),
        max_bytes=typing.cast("int", transfer_metadata_payload["max_bytes"]),
        total_elements=typing.cast("int", transfer_metadata_payload["total_elements"]),
    )


def build_stage_timing_recorder(
    stage_timing_path: pathlib.Path | None,
    *,
    force: bool,
) -> StageTimingRecorder | None:
    """Create a diagnostic stage recorder when requested."""
    native_recorder = _core.NativeStageTimingRecorder.from_config(stage_timing_path is not None, force)
    if native_recorder is None:
        return None
    return StageTimingRecorder.from_native_recorder(native_recorder)


def native_final_timing_output_policy() -> _core.NativeFinalTimingOutputPolicy:
    """Build the native final timing output policy handle."""
    return _core.NativeFinalTimingOutputPolicy()


def resolve_final_timing_output_context(
    diagnostics_stage_timing_path: pathlib.Path | None,
    telemetry_session: object | None,
) -> FinalTimingOutputContext:
    """Resolve final timing output paths through the native runtime policy."""
    native_context = native_final_timing_output_policy().resolve_final_timing_output_context(
        None if diagnostics_stage_timing_path is None else str(diagnostics_stage_timing_path),
        telemetry_session,
    )
    return FinalTimingOutputContext(
        stage_timing_path=path_from_native_context_value(native_context.stage_timing_path),
        profile_summary_path=path_from_native_context_value(native_context.profile_summary_path),
        run_id=native_context.run_id,
        force_stage_timing_recorder=native_context.force_stage_timing_recorder,
    )


def record_final_timing_outputs_write_started_diagnostic_event(
    stage_timing_path: pathlib.Path | None,
    profile_summary_path: pathlib.Path | None,
    run_id: str | None,
) -> None:
    """Record that final timing output writes are starting."""
    native_final_timing_output_policy().record_final_timing_outputs_write_started_diagnostic_event(
        None if stage_timing_path is None else str(stage_timing_path),
        None if profile_summary_path is None else str(profile_summary_path),
        run_id,
    )


def path_from_native_context_value(path_value: str | None) -> pathlib.Path | None:
    """Adapt a native optional path string into a Python path."""
    if path_value is None:
        return None
    return pathlib.Path(path_value)


def should_collect_exact_stage_timings(stage_timing_recorder: StageTimingRecorder | None) -> bool:
    """Return whether timing should force synchronized exact stage measurements."""
    return stage_timing_recorder is not None and stage_timing_recorder.should_collect_exact_stage_timings()


def write_final_timing_outputs(
    stage_timing_recorder: StageTimingRecorder | None,
    *,
    stage_timing_path: pathlib.Path | None,
    profile_summary_path: pathlib.Path | None,
    run_id: str | None,
) -> dict[str, bool]:
    """Persist all configured final timing outputs."""
    if stage_timing_recorder is None:
        return {"wrote_stage_timing_snapshot": False, "wrote_profile_summary": False}
    return stage_timing_recorder.write_final_timing_outputs(
        stage_timing_path=stage_timing_path,
        profile_summary_path=profile_summary_path,
        run_id=run_id,
    )


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
