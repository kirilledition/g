"""Native-driven REGENIE step 2 pipeline wrappers."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from g import _core, types
from g.compute import regenie2_binary, regenie2_binary_types, regenie2_linear, regenie2_linear_types
from g.io import output, source

ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED_ENVIRONMENT_VARIABLE = (
    "G_REGENIE2_ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED"
)


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


class RegeniePredictionSourceProtocol(typing.Protocol):
    """Native prediction source interface used by the JAX callbacks."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return already-aligned LOCO predictions for one chromosome."""
        ...


@dataclass(frozen=True)
class NativeBgenRunInput:
    """Sample-aligned inputs retained in native form for BGEN REGENIE step 2.

    Attributes:
        native_aligned_sample_data: Rust-owned aligned sample identifiers and matrices.
        sample_indices: BGEN sample indices for native chunk delivery.
        phenotype_vector: JAX phenotype vector.
        covariate_matrix: JAX design matrix.
        is_binary_trait: Whether the run is for a binary trait.

    """

    native_aligned_sample_data: _core.NativeAlignedSampleData
    sample_indices: npt.NDArray[np.int64]
    phenotype_vector: jax.Array
    covariate_matrix: jax.Array
    is_binary_trait: bool


@dataclass(frozen=True)
class StageTimingSnapshot:
    """Diagnostic stage timing snapshot for one native REGENIE step 2 run.

    Attributes:
        stage_totals_seconds: Total wall time per measured stage.
        stage_counts: Number of observations per measured stage.
        native_bgen_profile: Native BGEN profile counters from the run engine.
        binary_chunk_diagnostics: Binary score/Firth diagnostics per processed chunk.

    """

    stage_totals_seconds: dict[str, float]
    stage_counts: dict[str, int]
    native_bgen_profile: dict[str, int]
    binary_chunk_diagnostics: tuple[dict[str, int | float], ...]


class StageTimingRecorder:
    """Thread-safe diagnostic wall-time collector for profiling harnesses."""

    def __init__(self) -> None:
        """Initialize empty stage timing state."""
        self.stage_totals_seconds: dict[str, float] = {}
        self.stage_counts: dict[str, int] = {}
        self.native_bgen_profile: dict[str, int] = {}
        self.binary_chunk_diagnostics: list[dict[str, int | float]] = []
        self.lock = threading.Lock()

    def add_stage_duration(self, stage_name: str, duration_seconds: float) -> None:
        """Accumulate one measured duration."""
        with self.lock:
            self.stage_totals_seconds[stage_name] = self.stage_totals_seconds.get(stage_name, 0.0) + duration_seconds
            self.stage_counts[stage_name] = self.stage_counts.get(stage_name, 0) + 1

    def set_native_bgen_profile(self, profile_snapshot: dict[str, int]) -> None:
        """Store native BGEN profiling counters."""
        with self.lock:
            self.native_bgen_profile = dict(profile_snapshot)

    def add_binary_chunk_diagnostics(self, diagnostics: dict[str, int | float]) -> None:
        """Store diagnostic counters for one binary chunk."""
        with self.lock:
            self.binary_chunk_diagnostics.append(dict(diagnostics))

    def snapshot(self) -> StageTimingSnapshot:
        """Return an immutable copy of the current timings."""
        with self.lock:
            return StageTimingSnapshot(
                stage_totals_seconds=dict(self.stage_totals_seconds),
                stage_counts=dict(self.stage_counts),
                native_bgen_profile=dict(self.native_bgen_profile),
                binary_chunk_diagnostics=tuple(dict(diagnostics) for diagnostics in self.binary_chunk_diagnostics),
            )


def build_stage_timing_recorder_from_environment() -> StageTimingRecorder | None:
    """Create a diagnostic stage recorder when requested by the profiling harness."""
    if not os.environ.get("G_REGENIE2_STAGE_TIMINGS_JSON"):
        return None
    return StageTimingRecorder()


def assume_trusted_no_missing_diploid_validated() -> bool:
    """Return whether trusted BGEN validation should be treated as already completed."""
    raw_value = os.environ.get(ASSUME_TRUSTED_NO_MISSING_DIPLOID_VALIDATED_ENVIRONMENT_VARIABLE)
    if raw_value is None:
        return False
    return raw_value.lower() in {"1", "true", "yes", "on"}


def write_stage_timing_snapshot_from_environment(stage_timing_recorder: StageTimingRecorder | None) -> None:
    """Persist diagnostic stage timings when the profiling harness requests them."""
    if stage_timing_recorder is None:
        return
    stage_timing_path = os.environ.get("G_REGENIE2_STAGE_TIMINGS_JSON")
    if not stage_timing_path:
        return
    snapshot = stage_timing_recorder.snapshot()
    payload = {
        "stage_totals_seconds": snapshot.stage_totals_seconds,
        "stage_counts": snapshot.stage_counts,
        "native_bgen_profile": snapshot.native_bgen_profile,
        "binary_chunk_diagnostics": snapshot.binary_chunk_diagnostics,
    }
    Path(stage_timing_path).parent.mkdir(parents=True, exist_ok=True)
    Path(stage_timing_path).write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def record_stage_duration(
    stage_timing_recorder: StageTimingRecorder | None,
    stage_name: str,
    start_time: float,
) -> None:
    """Record elapsed wall time for a stage when diagnostics are active."""
    if stage_timing_recorder is None:
        return
    stage_timing_recorder.add_stage_duration(stage_name, time.perf_counter() - start_time)


def block_until_ready(value: typing.Any) -> None:
    """Synchronize a JAX value when it supports readiness blocking."""
    block_until_ready_method = getattr(value, "block_until_ready", None)
    if callable(block_until_ready_method):
        block_until_ready_method()


def record_binary_chunk_diagnostics(
    *,
    stage_timing_recorder: StageTimingRecorder | None,
    result: regenie2_binary_types.Regenie2BinaryChunkResult,
) -> None:
    """Record binary candidate and Firth diagnostics for one chunk."""
    if stage_timing_recorder is None:
        return
    firth_iteration_count = result.firth_iteration_count
    firth_attempt_mask = firth_iteration_count > 0
    firth_candidate_count = jnp.sum(firth_attempt_mask, dtype=jnp.int32)
    finite_iteration_count = jnp.where(firth_attempt_mask, firth_iteration_count, jnp.asarray(0, dtype=jnp.int32))
    sorted_active_iteration_count = jnp.sort(
        jnp.where(firth_attempt_mask, firth_iteration_count, np.iinfo(np.int32).max)
    )
    median_iteration_index = jnp.maximum((firth_candidate_count - 1) // 2, 0)
    diagnostics = jax.device_get(
        {
            "score_test_candidate_count": jnp.sum(
                (result.extra_code == regenie2_binary.EXTRA_CODE_FIRTH)
                | (result.extra_code == regenie2_binary.EXTRA_CODE_SPA)
                | (result.extra_code == regenie2_binary.EXTRA_CODE_TEST_FAIL),
                dtype=jnp.int32,
            ),
            "firth_candidate_count": firth_candidate_count,
            "firth_iteration_min": jnp.where(
                firth_candidate_count > 0,
                sorted_active_iteration_count[0],
                jnp.asarray(0, dtype=jnp.int32),
            ),
            "firth_iteration_median": jnp.where(
                firth_candidate_count > 0,
                sorted_active_iteration_count[median_iteration_index],
                jnp.asarray(0, dtype=jnp.int32),
            ),
            "firth_iteration_max": jnp.max(finite_iteration_count),
            "firth_converged_count": jnp.sum(result.extra_code == regenie2_binary.EXTRA_CODE_FIRTH, dtype=jnp.int32),
            "firth_failed_count": jnp.sum(result.extra_code == regenie2_binary.EXTRA_CODE_TEST_FAIL, dtype=jnp.int32),
            "firth_numerical_failure_count": jnp.sum(
                result.firth_failure_code == regenie2_binary.FIRTH_FAILURE_NUMERICAL,
                dtype=jnp.int32,
            ),
            "firth_max_iteration_failure_count": jnp.sum(
                result.firth_failure_code == regenie2_binary.FIRTH_FAILURE_MAX_ITERATIONS,
                dtype=jnp.int32,
            ),
            "firth_invalid_statistic_failure_count": jnp.sum(
                result.firth_failure_code == regenie2_binary.FIRTH_FAILURE_INVALID_STATISTIC,
                dtype=jnp.int32,
            ),
        }
    )
    stage_timing_recorder.add_binary_chunk_diagnostics(
        {key: int(value) if key != "firth_iteration_median" else float(value) for key, value in diagnostics.items()}
    )


def put_genotype_matrix_on_device(
    genotype_matrix: jax.Array | npt.NDArray[np.float32],
    stage_timing_recorder: StageTimingRecorder | None,
) -> jax.Array:
    """Transfer a genotype chunk to the active JAX device with optional timing."""
    start_time = time.perf_counter()
    genotype_device_array = jax.device_put(genotype_matrix)
    if stage_timing_recorder is not None:
        block_until_ready(genotype_device_array)
    record_stage_duration(stage_timing_recorder, "host_to_device_transfer", start_time)
    return genotype_device_array


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
    stage_timing_recorder: StageTimingRecorder | None,
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
    record_stage_duration(stage_timing_recorder, "device_to_host_materialization", materialization_start_time)

    write_start_time = time.perf_counter()
    writer_session.write_regenie2_native_chunk(
        metadata=metadata,
        chunk_stats=chunk_stats,
        beta=host_values["beta"],
        standard_error=host_values["standard_error"],
        chi_squared=host_values["chi_squared"],
        log10_p_value=host_values["log10_p_value"],
        extra_code=host_values["extra_code"],
    )
    record_stage_duration(stage_timing_recorder, "output_write", write_start_time)


def build_native_bgen_run_input(
    native_aligned_sample_data: _core.NativeAlignedSampleData,
) -> NativeBgenRunInput:
    """Build Python/JAX views over Rust-owned aligned sample data."""
    return NativeBgenRunInput(
        native_aligned_sample_data=native_aligned_sample_data,
        sample_indices=np.ascontiguousarray(native_aligned_sample_data.sample_indices, dtype=np.int64),
        phenotype_vector=jnp.asarray(native_aligned_sample_data.phenotype_vector, dtype=jnp.float32),
        covariate_matrix=jnp.asarray(native_aligned_sample_data.covariate_matrix, dtype=jnp.float32),
        is_binary_trait=native_aligned_sample_data.is_binary_trait,
    )


def load_native_aligned_sample_data_from_individual_identifier_table(
    *,
    sample_table: typing.Any,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
) -> _core.NativeAlignedSampleData:
    """Load Rust-owned aligned sample data from explicit sample identifiers."""
    return _core.align_sample_data(
        np.ascontiguousarray(sample_table.get_column("sample_index").to_numpy(), dtype=np.int64),
        typing.cast("list[str]", sample_table.get_column("family_identifier").to_list()),
        typing.cast("list[str]", sample_table.get_column("individual_identifier").to_list()),
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
    )


def load_native_aligned_sample_data_from_sample_file(
    *,
    sample_path: Path,
    expected_sample_count: int,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
) -> _core.NativeAlignedSampleData:
    """Load Rust-owned aligned sample data through Oxford sample-file parsing."""
    return _core.align_sample_data_from_sample_file(
        str(sample_path),
        expected_sample_count,
        str(phenotype_path),
        phenotype_name,
        str(covariate_path) if covariate_path is not None else None,
        list(covariate_names) if covariate_names is not None else None,
        is_binary_trait,
    )


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
        stage_timing_recorder: StageTimingRecorder | None = None,
    ) -> None:
        """Initialize shared native callback state."""
        self.processed_chunk_count = 0
        self.stage_timing_recorder = stage_timing_recorder
        self.dosage_queue_depth = max(1, staging_depth)
        self.dosage_buffer_limit = self.dosage_queue_depth + 1
        self.dosage_queue: queue.Queue[
            PreprocessedDosageChunkWorkItem | PreprocessedVariantMajorDosageChunkWorkItem | None
        ] = queue.Queue(maxsize=self.dosage_queue_depth)
        self.free_dosage_buffers: queue.Queue[npt.NDArray[np.float32]] = queue.Queue(maxsize=self.dosage_buffer_limit)
        self.dosage_buffer_count = 0
        self.worker_error: BaseException | None = None
        self.worker_thread = threading.Thread(
            target=self.consume_dosage_chunks,
            name=worker_name,
            daemon=True,
        )
        self.worker_thread.start()

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
                    self.release_dosage_buffer(work_item.genotype_matrix_by_variant)
                    continue
                if isinstance(work_item, PreprocessedDosageChunkWorkItem):
                    self.compute_preprocessed_chunk(
                        variant_metadata=work_item.metadata,
                        genotype_matrix=work_item.genotype_matrix,
                        chunk_stats=work_item.chunk_stats,
                    )
                    self.processed_chunk_count += 1
                    self.release_dosage_buffer(work_item.genotype_matrix)
                    continue
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

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
                record_stage_duration(self.stage_timing_recorder, "callback_queue_put", put_start_time)
                return
            except queue.Full:
                record_stage_duration(self.stage_timing_recorder, "callback_queue_producer_blocking", put_start_time)
                continue

    def raise_worker_error_if_present(self) -> None:
        """Raise an asynchronous worker failure on the producer thread."""
        if self.worker_error is not None:
            message = f"native pipeline callback worker failed: {self.worker_error}"
            raise RuntimeError(message) from self.worker_error

    def finish(self) -> None:
        """Wait until all queued JAX work has been written."""
        self.put_dosage_work_item(None)
        self.worker_thread.join()
        self.raise_worker_error_if_present()

    def abort(self) -> None:
        """Stop the worker after an upstream failure."""
        with contextlib.suppress(queue.Full):
            self.dosage_queue.put_nowait(None)

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


class LinearRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for quantitative traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInput,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        staging_depth: int = 1,
        stage_timing_recorder: StageTimingRecorder | None = None,
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
        self.current_chromosome_state: regenie2_linear_types.Regenie2LinearChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-linear-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and write it."""
        result = self.compute_linear_result(variant_metadata=variant_metadata, genotype_matrix=genotype_matrix)
        write_regenie2_native_chunk_with_optional_timing(
            writer_session=self.writer_session,
            metadata=variant_metadata,
            chunk_stats=chunk_stats,
            beta=result.beta,
            standard_error=result.standard_error,
            chi_squared=result.chi_squared,
            log10_p_value=result.log10_p_value,
            extra_code=None,
            stage_timing_recorder=self.stage_timing_recorder,
        )

    def compute_linear_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
    ) -> regenie2_linear_types.Regenie2LinearChunkResult:
        """Compute quantitative REGENIE step 2 statistics for one chunk."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome != self.current_chromosome:
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
            record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
            self.current_chromosome = chromosome
        assert self.current_chromosome_state is not None

        genotype_device_array = put_genotype_matrix_on_device(genotype_matrix, self.stage_timing_recorder)
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix=genotype_device_array,
        )
        block_until_ready(result.log10_p_value)
        record_stage_duration(self.stage_timing_recorder, "jax_compute", compute_start_time)
        return result


class BinaryRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for binary traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInput,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        correction_plan: types.BinaryCorrectionPlan,
        staging_depth: int = 1,
        stage_timing_recorder: StageTimingRecorder | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.correction_plan = correction_plan
        self.regenie_state = regenie2_binary.prepare_regenie2_binary_state(
            covariate_matrix=run_input.covariate_matrix,
            phenotype_vector=run_input.phenotype_vector,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_binary_types.Regenie2BinaryChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-binary-callback",
            staging_depth=staging_depth,
            stage_timing_recorder=stage_timing_recorder,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and write it."""
        result = self.compute_binary_result(
            variant_metadata=variant_metadata,
            genotype_matrix=genotype_matrix,
            sparse_candidate_mask=jax.device_put(chunk_stats.is_sparse_candidate),
        )
        write_regenie2_native_chunk_with_optional_timing(
            writer_session=self.writer_session,
            metadata=variant_metadata,
            chunk_stats=chunk_stats,
            beta=result.beta,
            standard_error=result.standard_error,
            chi_squared=result.chi_squared,
            log10_p_value=result.log10_p_value,
            extra_code=result.extra_code,
            stage_timing_recorder=self.stage_timing_recorder,
        )

    def compute_binary_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        sparse_candidate_mask: jax.Array | None = None,
    ) -> regenie2_binary_types.Regenie2BinaryChunkResult:
        """Compute binary REGENIE step 2 statistics for one chunk."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome != self.current_chromosome:
            chromosome_start_time = time.perf_counter()
            loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
            self.current_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
                self.regenie_state,
                loco_offset,
            )
            chromosome_ready_value = getattr(
                self.current_chromosome_state,
                "fitted_probability",
                self.current_chromosome_state,
            )
            block_until_ready(chromosome_ready_value)
            record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
            self.current_chromosome = chromosome
        assert self.current_chromosome_state is not None

        genotype_device_array = put_genotype_matrix_on_device(genotype_matrix, self.stage_timing_recorder)
        compute_start_time = time.perf_counter()
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix=genotype_device_array,
            correction_plan=self.correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
        )
        block_until_ready(result.log10_p_value)
        record_stage_duration(self.stage_timing_recorder, "jax_compute", compute_start_time)
        record_binary_chunk_diagnostics(stage_timing_recorder=self.stage_timing_recorder, result=result)
        return result

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed variant-major chunk and write it."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome != self.current_chromosome:
            chromosome_start_time = time.perf_counter()
            loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
            self.current_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
                self.regenie_state,
                loco_offset,
            )
            chromosome_ready_value = getattr(
                self.current_chromosome_state,
                "fitted_probability",
                self.current_chromosome_state,
            )
            block_until_ready(chromosome_ready_value)
            record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
            self.current_chromosome = chromosome
        assert self.current_chromosome_state is not None

        genotype_device_array = put_genotype_matrix_on_device(genotype_matrix_by_variant, self.stage_timing_recorder)
        compute_start_time = time.perf_counter()
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix=jnp.transpose(genotype_device_array),
            correction_plan=self.correction_plan,
            sparse_candidate_mask=jax.device_put(chunk_stats.is_sparse_candidate),
        )
        block_until_ready(result.log10_p_value)
        record_stage_duration(self.stage_timing_recorder, "jax_compute", compute_start_time)
        record_binary_chunk_diagnostics(stage_timing_recorder=self.stage_timing_recorder, result=result)
        write_regenie2_native_chunk_with_optional_timing(
            writer_session=self.writer_session,
            metadata=variant_metadata,
            chunk_stats=chunk_stats,
            beta=result.beta,
            standard_error=result.standard_error,
            chi_squared=result.chi_squared,
            log10_p_value=result.log10_p_value,
            extra_code=result.extra_code,
            stage_timing_recorder=self.stage_timing_recorder,
        )


@dataclass(frozen=True)
class WarmCacheShape:
    """One genotype matrix shape warmed for the JAX compilation cache."""

    sample_count: int
    variant_count: int


@dataclass(frozen=True)
class WarmCacheReport:
    """Summary of warmed REGENIE step 2 JAX cache entries."""

    warmed_shapes: tuple[WarmCacheShape, ...]


def build_warm_cache_shapes(
    *,
    engine: _core.Regenie2RunEngine,
    chunk_size: int,
    variant_limit: int | None,
    sample_count: int,
) -> tuple[WarmCacheShape, ...]:
    """Build the full and tail chunk shapes that should be warmed."""
    chunk_specs = _core.plan_genotype_chunks(
        engine.variant_count,
        chunk_size,
        engine.chromosome_boundary_indices(),
        variant_limit=variant_limit,
        committed_chunk_identifiers=None,
    )
    variant_counts = []
    for chunk_spec in chunk_specs:
        variant_count = int(chunk_spec.variant_stop_index - chunk_spec.variant_start_index)
        if variant_count > 0 and variant_count not in variant_counts:
            variant_counts.append(variant_count)
    variant_counts.sort(reverse=True)
    return tuple(
        WarmCacheShape(sample_count=sample_count, variant_count=variant_count) for variant_count in variant_counts[:2]
    )


def build_synthetic_genotype_matrix(
    *,
    phenotype_vector: jax.Array,
    variant_count: int,
    is_binary_trait: bool,
) -> jax.Array:
    """Build deterministic genotype inputs for cache warming."""
    if is_binary_trait:
        genotype_vector = jnp.asarray(phenotype_vector, dtype=jnp.float32) * 2.0
    else:
        sample_index = jnp.arange(phenotype_vector.shape[0], dtype=jnp.float32)
        genotype_vector = jnp.mod(sample_index, 3.0)
        genotype_vector = genotype_vector - jnp.mean(genotype_vector)
    return jnp.tile(genotype_vector[:, None], (1, variant_count))


def warm_regenie2_linear_bgen_cache(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool = False,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for quantitative REGENIE step 2."""
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
    )
    chromosome = first_engine_chromosome(engine)
    regenie_state = regenie2_linear.prepare_regenie2_linear_state(
        covariate_matrix=run_input.covariate_matrix,
        phenotype_vector=run_input.phenotype_vector,
    )
    chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
        regenie_state,
        jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix = build_synthetic_genotype_matrix(
            phenotype_vector=run_input.phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=False,
        )
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
        )
        block_until_ready(result.log10_p_value)
    return WarmCacheReport(warmed_shapes=shapes)


def warm_regenie2_binary_bgen_cache(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    correction_plan: types.BinaryCorrectionPlan,
    trusted_no_missing_diploid: bool = False,
) -> WarmCacheReport:
    """Warm full and tail JAX compilation-cache shapes for binary REGENIE step 2."""
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
    )
    chromosome = first_engine_chromosome(engine)
    regenie_state = regenie2_binary.prepare_regenie2_binary_state(
        covariate_matrix=run_input.covariate_matrix,
        phenotype_vector=run_input.phenotype_vector,
    )
    chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
        regenie_state,
        jax.device_put(prediction_source.get_chromosome_predictions(chromosome)),
    )
    shapes = build_warm_cache_shapes(
        engine=engine,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        sample_count=int(run_input.sample_indices.shape[0]),
    )
    for shape in shapes:
        genotype_matrix = build_synthetic_genotype_matrix(
            phenotype_vector=run_input.phenotype_vector,
            variant_count=shape.variant_count,
            is_binary_trait=True,
        )
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_matrix,
            correction_plan=correction_plan,
        )
        block_until_ready(result.log10_p_value)
    return WarmCacheReport(warmed_shapes=shapes)


def first_engine_chromosome(engine: _core.Regenie2RunEngine) -> str:
    """Return the first chromosome label from the native BGEN engine."""
    chromosome_values, _, _, _, _ = engine.variant_metadata_slice(0, 1)
    if not chromosome_values:
        message = "Cannot warm REGENIE step 2 cache for an empty BGEN dataset."
        raise ValueError(message)
    return chromosome_values[0]


def run_regenie2_linear_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int = 1,
    committed_chunk_identifiers: set[int] | None = None,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    trusted_no_missing_diploid: bool = False,
    stage_timing_recorder: StageTimingRecorder | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder_from_environment()
    engine_start_time = time.perf_counter()
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    alignment_start_time = time.perf_counter()
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
    writer_start_time = time.perf_counter()
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
    )
    record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    prediction_start_time = time.perf_counter()
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
    )
    record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    callback = LinearRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        staging_depth=staging_depth,
        stage_timing_recorder=stage_timing_recorder,
    )
    return run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
    )


def run_regenie2_binary_bgen_pipeline(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    phenotype_path: Path,
    phenotype_name: str,
    prediction_list_path: Path,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    chunk_size: int,
    variant_limit: int | None,
    output_run_paths: output.OutputRunPaths,
    staging_depth: int = 1,
    committed_chunk_identifiers: set[int] | None = None,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    trusted_no_missing_diploid: bool = False,
    correction_plan: types.BinaryCorrectionPlan = types.BinaryCorrectionPlan(),
    stage_timing_recorder: StageTimingRecorder | None = None,
) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    stage_timing_recorder = stage_timing_recorder or build_stage_timing_recorder_from_environment()
    use_variant_major = trusted_no_missing_diploid
    engine_start_time = time.perf_counter()
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    record_stage_duration(stage_timing_recorder, "bgen_engine_open_index_setup", engine_start_time)
    alignment_start_time = time.perf_counter()
    run_input = load_native_bgen_run_input(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    record_stage_duration(stage_timing_recorder, "sample_phenotype_covariate_alignment", alignment_start_time)
    writer_start_time = time.perf_counter()
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_BINARY,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
    )
    record_stage_duration(stage_timing_recorder, "output_writer_preparation", writer_start_time)
    prediction_start_time = time.perf_counter()
    prediction_source = build_regenie_prediction_source(
        prediction_list_path=prediction_list_path,
        phenotype_name=phenotype_name,
        run_input=run_input,
    )
    record_stage_duration(stage_timing_recorder, "prediction_source_load", prediction_start_time)
    callback = BinaryRegenie2PipelineCallback(
        run_input=run_input,
        prediction_source=prediction_source,
        writer_session=writer_session,
        correction_plan=correction_plan,
        staging_depth=staging_depth,
        stage_timing_recorder=stage_timing_recorder,
    )
    return run_bgen_engine_with_callback(
        engine=engine,
        run_input=run_input,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_session=writer_session,
        callback=callback,
        stage_timing_recorder=stage_timing_recorder,
        variant_major_dosage=use_variant_major,
    )


def load_native_bgen_run_input(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
) -> NativeBgenRunInput:
    """Load native-aligned samples and JAX compute inputs for a native BGEN run."""
    source.validate_genotype_source_config(genotype_source_config)
    resolved_sample_path = source.resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    if resolved_sample_path is not None:
        native_aligned_sample_data = load_native_aligned_sample_data_from_sample_file(
            sample_path=resolved_sample_path,
            expected_sample_count=engine.sample_count,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=is_binary_trait,
        )
        return build_native_bgen_run_input(native_aligned_sample_data)
    if engine.contains_embedded_samples:
        sample_table = source.build_sample_identifier_table(np.asarray(engine.sample_identifiers(), dtype=np.str_))
        native_aligned_sample_data = load_native_aligned_sample_data_from_individual_identifier_table(
            sample_table=sample_table,
            phenotype_path=phenotype_path,
            phenotype_name=phenotype_name,
            covariate_path=covariate_path,
            covariate_names=covariate_names,
            is_binary_trait=is_binary_trait,
        )
        return build_native_bgen_run_input(native_aligned_sample_data)
    message = "BGEN file does not contain samples and no .sample file was found."
    raise ValueError(message)


def build_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_name: str,
    run_input: NativeBgenRunInput,
) -> _core.RegeniePredictionSource:
    """Load Rust-owned REGENIE step 1 predictions aligned to the run samples."""
    return _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        phenotype_name,
        run_input.native_aligned_sample_data,
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
    trusted_no_missing_diploid: bool = False,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine once for alignment and chunk delivery."""
    engine = _core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=trusted_no_missing_diploid,
    )
    if trusted_no_missing_diploid and not assume_trusted_no_missing_diploid_validated():
        engine.validate_trusted_no_missing_diploid()
    return engine


def run_bgen_engine_with_callback(
    *,
    engine: _core.Regenie2RunEngine,
    run_input: NativeBgenRunInput,
    committed_chunk_identifiers: set[int] | None,
    writer_session: typing.Any,
    callback: object,
    stage_timing_recorder: StageTimingRecorder | None,
    variant_major_dosage: bool = False,
) -> Path | None:
    """Run native BGEN chunk delivery and close the output writer."""
    try:
        if stage_timing_recorder is not None:
            engine.reset_profile()
        engine_delivery_start_time = time.perf_counter()
        sample_indices = run_input.sample_indices
        committed_chunk_identifier_list = sorted(committed_chunk_identifiers or set())
        if variant_major_dosage:
            engine.run_bgen_variant_major_dosage_buffered_chunks(
                sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        else:
            engine.run_bgen_dosage_buffered_chunks(
                sample_indices,
                callback,
                committed_chunk_identifiers=committed_chunk_identifier_list,
            )
        record_stage_duration(stage_timing_recorder, "native_engine_delivery", engine_delivery_start_time)
        if stage_timing_recorder is not None:
            stage_timing_recorder.set_native_bgen_profile(engine.profile_snapshot())
        callback_finish_start_time = time.perf_counter()
        typing.cast("typing.Any", callback).finish()
        record_stage_duration(stage_timing_recorder, "callback_drain", callback_finish_start_time)
        writer_finish_start_time = time.perf_counter()
        final_parquet_path = writer_session.finish()
        record_stage_duration(stage_timing_recorder, "writer_finish_and_parquet_finalization", writer_finish_start_time)
    except Exception:
        abort_callback = getattr(callback, "abort", None)
        if callable(abort_callback):
            abort_callback()
        writer_session.abort()
        write_stage_timing_snapshot_from_environment(stage_timing_recorder)
        raise
    write_stage_timing_snapshot_from_environment(stage_timing_recorder)
    if final_parquet_path is None:
        return None
    return Path(final_parquet_path)
