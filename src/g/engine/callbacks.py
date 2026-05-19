"""Native BGEN callback helpers for REGENIE step 2 pipelines."""

from __future__ import annotations

import contextlib
import queue
import threading
import time
import typing
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import g._core as core
import g.compute.regenie2_binary as regenie2_binary
import g.compute.regenie2_binary_types as regenie2_binary_types
import g.compute.regenie2_linear as regenie2_linear
import g.compute.regenie2_linear_types as regenie2_linear_types
import g.engine.timing as timing
import g.types as g_types

StageTimingRecorder = timing.StageTimingRecorder
record_stage_duration = timing.record_stage_duration


@dataclass(frozen=True)
class PreprocessedDosageChunkWorkItem:
    """One native-preprocessed dosage chunk staged for asynchronous JAX compute."""

    metadata: typing.Any
    genotype_matrix: npt.NDArray[np.float32]
    chunk_stats: core.ChunkStats


@dataclass(frozen=True)
class PreprocessedVariantMajorDosageChunkWorkItem:
    """One native-preprocessed variant-major dosage chunk staged for JAX compute."""

    metadata: typing.Any
    genotype_matrix_by_variant: npt.NDArray[np.float32]
    chunk_stats: core.ChunkStats


class NativeBgenRunInputProtocol(typing.Protocol):
    """Run input fields required by callback compute initialization."""

    phenotype_vector: jax.Array
    covariate_matrix: jax.Array


class RegeniePredictionSourceProtocol(typing.Protocol):
    """Native prediction source interface used by the JAX callbacks."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return already-aligned LOCO predictions for one chromosome."""
        ...


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
    metadata: core.VariantMetadata,
    chunk_stats: core.ChunkStats,
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
        variant_metadata: core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed chunk and write it."""
        raise NotImplementedError

    def compute_preprocessed_variant_major_chunk(
        self,
        *,
        variant_metadata: core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: core.ChunkStats,
    ) -> None:
        """Compute one Rust-preprocessed variant-major chunk and write it."""
        raise NotImplementedError

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: core.VariantMetadata,
        genotype_matrix: npt.NDArray[np.float32],
        chunk_stats: core.ChunkStats,
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
        metadata: core.VariantMetadata,
        genotype_matrix_by_variant: npt.NDArray[np.float32],
        chunk_stats: core.ChunkStats,
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
        run_input: NativeBgenRunInputProtocol,
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
        variant_metadata: core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: core.ChunkStats,
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
        run_input: NativeBgenRunInputProtocol,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        correction_plan: g_types.BinaryCorrectionPlan,
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
        variant_metadata: core.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        chunk_stats: core.ChunkStats,
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
        variant_metadata: core.VariantMetadata,
        genotype_matrix_by_variant: jax.Array | npt.NDArray[np.float32],
        chunk_stats: core.ChunkStats,
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
