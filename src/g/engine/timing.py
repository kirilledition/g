"""Timing diagnostics for REGENIE step 2 engine runs."""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass

from g import _core

if typing.TYPE_CHECKING:
    import pathlib

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


def should_collect_exact_stage_timings(stage_timing_recorder: StageTimingRecorder | None) -> bool:
    """Return whether timing should force synchronized exact stage measurements."""
    return (
        stage_timing_recorder is not None and stage_timing_recorder.native_recorder.should_collect_exact_stage_timings()
    )


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
    return dict(
        typing.cast(
            "typing.Mapping[str, bool]",
            stage_timing_recorder.native_recorder.write_final_timing_outputs(
                None if stage_timing_path is None else str(stage_timing_path),
                None if profile_summary_path is None else str(profile_summary_path),
                run_id,
            ),
        )
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
