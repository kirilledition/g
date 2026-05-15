"""Native-driven REGENIE step 2 pipeline wrappers."""

from __future__ import annotations

import contextlib
import queue
import threading
import typing
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import numpy.typing as npt

from g import _core, models, types
from g.compute import regenie2_binary, regenie2_linear
from g.engine import types as engine_types
from g.io import bgen, genotype_processing, output, source


@dataclass(frozen=True)
class DosageChunkWorkItem:
    """One raw dosage chunk staged for asynchronous JAX processing."""

    metadata: models.VariantMetadata
    genotype_matrix: npt.NDArray[np.float32]


class RegeniePredictionSourceProtocol(typing.Protocol):
    """Native prediction source interface used by the JAX callbacks."""

    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]:
        """Return already-aligned LOCO predictions for one chromosome."""
        ...


def build_variant_metadata(native_metadata: _core.VariantMetadata) -> models.VariantMetadata:
    """Convert native Rust metadata into the Python model used by compute/output code."""
    return models.VariantMetadata(
        variant_start_index=native_metadata.variant_start_index,
        variant_stop_index=native_metadata.variant_stop_index,
        chromosome=np.asarray(native_metadata.chromosome, dtype=np.str_),
        variant_identifiers=np.asarray(native_metadata.variant_identifiers, dtype=np.str_),
        position=np.asarray(native_metadata.position, dtype=np.int64),
        allele_one=np.asarray(native_metadata.allele_one, dtype=np.str_),
        allele_two=np.asarray(native_metadata.allele_two, dtype=np.str_),
    )


class LinearRegenie2PipelineCallback:
    """Compute/write callback used by the native BGEN pipeline for quantitative traits."""

    def __init__(
        self,
        aligned_sample_data: models.AlignedSampleData,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        staging_depth: int = 1,
    ) -> None:
        """Initialize the callback state."""
        self.aligned_sample_data = aligned_sample_data
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.regenie_state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: models.Regenie2LinearChromosomeState | None = None
        self.processed_chunk_count = 0
        self.dosage_queue_depth = max(1, staging_depth)
        self.dosage_buffer_limit = self.dosage_queue_depth + 1
        self.dosage_queue: queue.Queue[DosageChunkWorkItem | None] = queue.Queue(maxsize=self.dosage_queue_depth)
        self.free_dosage_buffers: queue.Queue[npt.NDArray[np.float32]] = queue.Queue(maxsize=self.dosage_buffer_limit)
        self.dosage_buffer_count = 0
        self.worker_error: BaseException | None = None
        self.worker_thread = threading.Thread(
            target=self.consume_dosage_chunks,
            name="regenie2-linear-callback",
            daemon=True,
        )
        self.worker_thread.start()

    def compute_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: npt.NDArray[np.float32],
        allele_one_frequency: npt.NDArray[np.float32],
        observation_count: npt.NDArray[np.int32],
    ) -> None:
        """Compute one Rust-provided chunk and write it through the native output sink."""
        variant_metadata = build_variant_metadata(metadata)
        self.compute_preprocessed_chunk(
            variant_metadata=variant_metadata,
            genotype_matrix=genotype_matrix,
            allele_one_frequency=allele_one_frequency,
            observation_count=observation_count,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: models.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        allele_one_frequency: jax.Array | npt.NDArray[np.float32],
        observation_count: jax.Array | npt.NDArray[np.int32],
    ) -> None:
        """Compute one already-preprocessed chunk and write it."""
        chromosome = str(variant_metadata.chromosome[0])
        if chromosome != self.current_chromosome:
            loco_predictions = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
            self.current_chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
                self.regenie_state,
                loco_predictions,
            )
            self.current_chromosome = chromosome
        assert self.current_chromosome_state is not None

        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix=jax.device_put(genotype_matrix),
        )
        output.write_regenie2_chunk(
            self.writer_session,
            build_chunk_accumulator(
                metadata=variant_metadata,
                allele_one_frequency=allele_one_frequency,
                observation_count=observation_count,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=None,
            ),
        )
        self.processed_chunk_count += 1

    def compute_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: npt.NDArray[np.float32],
    ) -> None:
        """Enqueue one Rust-provided dosage chunk for JAX processing."""
        self.put_dosage_work_item(
            DosageChunkWorkItem(metadata=build_variant_metadata(metadata), genotype_matrix=genotype_matrix)
        )

    def consume_dosage_chunks(self) -> None:
        """Consume queued dosage chunks and run JAX work in order."""
        try:
            while True:
                work_item = self.dosage_queue.get()
                if work_item is None:
                    return
                preprocessed_genotype_arrays = genotype_processing.preprocess_genotype_matrix_arrays(
                    jax.device_put(work_item.genotype_matrix)
                )
                self.compute_preprocessed_chunk(
                    variant_metadata=work_item.metadata,
                    genotype_matrix=preprocessed_genotype_arrays.genotypes,
                    allele_one_frequency=preprocessed_genotype_arrays.allele_one_frequency,
                    observation_count=preprocessed_genotype_arrays.observation_count,
                )
                self.release_dosage_buffer(work_item.genotype_matrix)
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

    def put_dosage_work_item(self, work_item: DosageChunkWorkItem | None) -> None:
        """Put work into the bounded worker queue while surfacing worker errors."""
        while True:
            self.raise_worker_error_if_present()
            try:
                self.dosage_queue.put(work_item, timeout=0.1)
                return
            except queue.Full:
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


class BinaryRegenie2PipelineCallback:
    """Compute/write callback used by the native BGEN pipeline for binary traits."""

    def __init__(
        self,
        aligned_sample_data: models.AlignedSampleData,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        correction: types.RegenieBinaryCorrection,
        staging_depth: int = 1,
    ) -> None:
        """Initialize the callback state."""
        self.aligned_sample_data = aligned_sample_data
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.correction = correction
        self.regenie_state = regenie2_binary.prepare_regenie2_binary_state(
            covariate_matrix=aligned_sample_data.covariate_matrix,
            phenotype_vector=aligned_sample_data.phenotype_vector,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: models.Regenie2BinaryChromosomeState | None = None
        self.processed_chunk_count = 0
        self.dosage_queue_depth = max(1, staging_depth)
        self.dosage_buffer_limit = self.dosage_queue_depth + 1
        self.dosage_queue: queue.Queue[DosageChunkWorkItem | None] = queue.Queue(maxsize=self.dosage_queue_depth)
        self.free_dosage_buffers: queue.Queue[npt.NDArray[np.float32]] = queue.Queue(maxsize=self.dosage_buffer_limit)
        self.dosage_buffer_count = 0
        self.worker_error: BaseException | None = None
        self.worker_thread = threading.Thread(
            target=self.consume_dosage_chunks,
            name="regenie2-binary-callback",
            daemon=True,
        )
        self.worker_thread.start()

    def compute_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: npt.NDArray[np.float32],
        allele_one_frequency: npt.NDArray[np.float32],
        observation_count: npt.NDArray[np.int32],
    ) -> None:
        """Compute one Rust-provided chunk and write it through the native output sink."""
        variant_metadata = build_variant_metadata(metadata)
        self.compute_preprocessed_chunk(
            variant_metadata=variant_metadata,
            genotype_matrix=genotype_matrix,
            allele_one_frequency=allele_one_frequency,
            observation_count=observation_count,
        )

    def compute_preprocessed_chunk(
        self,
        *,
        variant_metadata: models.VariantMetadata,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        allele_one_frequency: jax.Array | npt.NDArray[np.float32],
        observation_count: jax.Array | npt.NDArray[np.int32],
    ) -> None:
        """Compute one already-preprocessed chunk and write it."""
        chromosome = str(variant_metadata.chromosome[0])
        if chromosome != self.current_chromosome:
            loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
            self.current_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
                self.regenie_state,
                loco_offset,
            )
            self.current_chromosome = chromosome
        assert self.current_chromosome_state is not None

        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=self.current_chromosome_state,
            genotype_matrix=jax.device_put(genotype_matrix),
            correction=self.correction,
        )
        output.write_regenie2_chunk(
            self.writer_session,
            build_chunk_accumulator(
                metadata=variant_metadata,
                allele_one_frequency=allele_one_frequency,
                observation_count=observation_count,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=result.extra_code,
            ),
        )
        self.processed_chunk_count += 1

    def compute_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: npt.NDArray[np.float32],
    ) -> None:
        """Enqueue one Rust-provided dosage chunk for JAX processing."""
        self.put_dosage_work_item(
            DosageChunkWorkItem(metadata=build_variant_metadata(metadata), genotype_matrix=genotype_matrix)
        )

    def consume_dosage_chunks(self) -> None:
        """Consume queued dosage chunks and run JAX work in order."""
        try:
            while True:
                work_item = self.dosage_queue.get()
                if work_item is None:
                    return
                preprocessed_genotype_arrays = genotype_processing.preprocess_genotype_matrix_arrays(
                    jax.device_put(work_item.genotype_matrix)
                )
                self.compute_preprocessed_chunk(
                    variant_metadata=work_item.metadata,
                    genotype_matrix=preprocessed_genotype_arrays.genotypes,
                    allele_one_frequency=preprocessed_genotype_arrays.allele_one_frequency,
                    observation_count=preprocessed_genotype_arrays.observation_count,
                )
                self.release_dosage_buffer(work_item.genotype_matrix)
        except Exception as error:  # noqa: BLE001
            self.worker_error = error

    def put_dosage_work_item(self, work_item: DosageChunkWorkItem | None) -> None:
        """Put work into the bounded worker queue while surfacing worker errors."""
        while True:
            self.raise_worker_error_if_present()
            try:
                self.dosage_queue.put(work_item, timeout=0.1)
                return
            except queue.Full:
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


def build_chunk_accumulator(
    *,
    metadata: models.VariantMetadata,
    allele_one_frequency: jax.Array | npt.NDArray[np.float32],
    observation_count: jax.Array | npt.NDArray[np.int32],
    beta: jax.Array,
    standard_error: jax.Array,
    chi_squared: jax.Array,
    log10_p_value: jax.Array,
    extra_code: jax.Array | None,
) -> engine_types.Regenie2ChunkAccumulator:
    """Build one chunk accumulator from Rust-side metadata and JAX outputs."""
    return engine_types.Regenie2ChunkAccumulator(
        metadata=metadata,
        allele_one_frequency=jax.device_put(allele_one_frequency),
        observation_count=jax.device_put(observation_count),
        beta=beta,
        standard_error=standard_error,
        chi_squared=chi_squared,
        log10_p_value=log10_p_value,
        extra_code=extra_code,
    )


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
    prefetch_chunks: int = 1,
    committed_chunk_identifiers: set[int] | None = None,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
) -> Path | None:
    """Run the native BGEN pipeline for quantitative REGENIE step 2."""
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    )
    aligned_sample_data = load_bgen_aligned_sample_data(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=False,
    )
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_LINEAR,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
    )
    callback = LinearRegenie2PipelineCallback(
        aligned_sample_data=aligned_sample_data,
        prediction_source=build_regenie_prediction_source(
            prediction_list_path=prediction_list_path,
            phenotype_name=phenotype_name,
            aligned_sample_data=aligned_sample_data,
        ),
        writer_session=writer_session,
        staging_depth=prefetch_chunks,
    )
    return run_bgen_engine_with_callback(
        engine=engine,
        aligned_sample_data=aligned_sample_data,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_session=writer_session,
        callback=callback,
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
    prefetch_chunks: int = 1,
    committed_chunk_identifiers: set[int] | None = None,
    finalize_parquet: bool = False,
    writer_thread_count: int = output.DEFAULT_WRITER_THREAD_COUNT,
    writer_queue_depth: int = output.DEFAULT_WRITER_QUEUE_DEPTH,
    correction: types.RegenieBinaryCorrection = types.RegenieBinaryCorrection.FIRTH_APPROXIMATE,
) -> Path | None:
    """Run the native BGEN pipeline for binary REGENIE step 2."""
    engine = build_bgen_run_engine(
        genotype_source_config=genotype_source_config,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    )
    aligned_sample_data = load_bgen_aligned_sample_data(
        genotype_source_config=genotype_source_config,
        engine=engine,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=True,
    )
    writer_session = output.create_output_writer_session(
        output_run_paths,
        types.AssociationMode.REGENIE2_BINARY,
        writer_thread_count=writer_thread_count,
        writer_queue_depth=writer_queue_depth,
        finalize_parquet=finalize_parquet,
    )
    callback = BinaryRegenie2PipelineCallback(
        aligned_sample_data=aligned_sample_data,
        prediction_source=build_regenie_prediction_source(
            prediction_list_path=prediction_list_path,
            phenotype_name=phenotype_name,
            aligned_sample_data=aligned_sample_data,
        ),
        writer_session=writer_session,
        correction=correction,
        staging_depth=prefetch_chunks,
    )
    return run_bgen_engine_with_callback(
        engine=engine,
        aligned_sample_data=aligned_sample_data,
        committed_chunk_identifiers=committed_chunk_identifiers,
        writer_session=writer_session,
        callback=callback,
    )


def load_bgen_aligned_sample_data(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    engine: _core.Regenie2RunEngine,
    phenotype_path: Path,
    phenotype_name: str,
    covariate_path: Path | None,
    covariate_names: tuple[str, ...] | None,
    is_binary_trait: bool,
) -> models.AlignedSampleData:
    """Load aligned samples for the native BGEN pipeline."""
    if genotype_source_config.source_format != types.GenotypeSourceFormat.BGEN:
        message = "The native pipeline currently supports BGEN genotype sources only."
        raise ValueError(message)
    resolved_sample_path = bgen.resolve_bgen_sample_path(
        genotype_source_config.source_path,
        genotype_source_config.sample_path,
    )
    if resolved_sample_path is not None:
        sample_table = bgen.load_sample_identifier_table(resolved_sample_path)
        if sample_table.height != engine.sample_count:
            message = (
                f"Expect number of samples in file to match BGEN sample count. "
                f"Sample file '{resolved_sample_path}' contains {sample_table.height} rows, "
                f"but '{genotype_source_config.source_path}' contains {engine.sample_count} samples."
            )
            raise ValueError(message)
    elif engine.contains_embedded_samples:
        sample_table = bgen.build_sample_identifier_table(np.asarray(engine.sample_identifiers(), dtype=np.str_))
    else:
        message = "BGEN file does not contain samples and no .sample file was found."
        raise ValueError(message)
    return source.load_aligned_sample_data_from_individual_identifier_table(
        sample_table=sample_table,
        phenotype_path=phenotype_path,
        phenotype_name=phenotype_name,
        covariate_path=covariate_path,
        covariate_names=covariate_names,
        is_binary_trait=is_binary_trait,
    )


def build_regenie_prediction_source(
    *,
    prediction_list_path: Path,
    phenotype_name: str,
    aligned_sample_data: models.AlignedSampleData,
) -> _core.RegeniePredictionSource:
    """Load Rust-owned REGENIE step 1 predictions aligned to the run samples."""
    sample_family_identifiers = typing.cast("list[str]", aligned_sample_data.family_identifiers.tolist())
    sample_individual_identifiers = typing.cast("list[str]", aligned_sample_data.individual_identifiers.tolist())
    return _core.RegeniePredictionSource(
        str(prediction_list_path),
        phenotype_name,
        sample_family_identifiers,
        sample_individual_identifiers,
    )


def build_bgen_run_engine(
    *,
    genotype_source_config: source.GenotypeSourceConfig,
    chunk_size: int,
    variant_limit: int | None,
) -> _core.Regenie2RunEngine:
    """Open the native BGEN run engine once for alignment and chunk delivery."""
    return _core.Regenie2RunEngine(
        str(genotype_source_config.source_path),
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    )


def run_bgen_engine_with_callback(
    *,
    engine: _core.Regenie2RunEngine,
    aligned_sample_data: models.AlignedSampleData,
    committed_chunk_identifiers: set[int] | None,
    writer_session: typing.Any,
    callback: object,
) -> Path | None:
    """Run native BGEN chunk delivery and close the output writer."""
    try:
        engine.run_bgen_dosage_buffered_chunks(
            np.ascontiguousarray(aligned_sample_data.sample_indices, dtype=np.int64),
            callback,
            committed_chunk_identifiers=sorted(committed_chunk_identifiers or set()),
        )
        typing.cast("typing.Any", callback).finish()
        final_parquet_path = writer_session.finish()
    except Exception:
        abort_callback = getattr(callback, "abort", None)
        if callable(abort_callback):
            abort_callback()
        writer_session.abort()
        raise
    if final_parquet_path is None:
        return None
    return Path(final_parquet_path)
