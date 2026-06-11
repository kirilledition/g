"""Linear REGENIE step 2 callback implementations."""

from __future__ import annotations

import time
import typing

import jax

import g.engine.callbacks.diagnostics as diagnostics
import g.engine.callbacks.runtime as runtime
import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
import g.engine.callbacks.writers as writers
from g import _core, types
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.engine import telemetry, timing

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

HostGenotypeBuffer = shared.HostGenotypeBuffer
NativeBgenRunInputProtocol = shared.NativeBgenRunInputProtocol
NativeBgenMultiRunInputProtocol = shared.NativeBgenMultiRunInputProtocol
RegeniePredictionSourceProtocol = shared.RegeniePredictionSourceProtocol
MultiRegeniePredictionSourceProtocol = shared.MultiRegeniePredictionSourceProtocol
Regenie2ResultWriteWorkItem = shared.Regenie2ResultWriteWorkItem
Regenie2MultiResultWriteWorkItem = shared.Regenie2MultiResultWriteWorkItem
NativeBgenCallbackRunner = runtime.NativeBgenCallbackRunner
require_current_chromosome_state = runtime.require_current_chromosome_state
put_compute_array_on_device = transfers.put_compute_array_on_device
put_genotype_matrix_on_device = transfers.put_genotype_matrix_on_device
put_chunk_array_on_device = transfers.put_chunk_array_on_device
get_linear_chunk_stats_arrays = transfers.get_linear_chunk_stats_arrays
block_compute_result_for_timing = transfers.block_compute_result_for_timing
write_regenie2_multi_native_chunk_with_optional_timing = writers.write_regenie2_multi_native_chunk_with_optional_timing
block_until_ready = diagnostics.block_until_ready
get_metadata_chromosome = shared.get_metadata_chromosome


class LinearRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for quantitative traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInputProtocol,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        staging_depth: int = 1,
        result_in_flight_limit: int | None = None,
        dosage_buffer_limit: int | None = None,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
        output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.score_dtype = score_dtype
        self.linear_numerical_config = linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_vector = put_compute_array_on_device(run_input.phenotype_vector)
        self.regenie_state = regenie2_linear.prepare_regenie2_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_linear.Regenie2LinearChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-linear-callback",
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            output_statistic_dtype=output_statistic_dtype,
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

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one packed8 chunk and enqueue its result for writing."""
        host_packed_buffer = self.get_releasable_dosage_buffer(packed_probability_pairs_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )

            packed_device_array = put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_observation_count = put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_linear_chunk_packed8_donating_inputs(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_device_array,
                genotype_dosage_sum=genotype_dosage_sum,
                genotype_observation_count=genotype_observation_count,
                genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_linear_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_packed_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_packed_buffer is not None:
                self.release_dosage_buffer(host_packed_buffer)
            self.release_result_in_flight_slot()
            raise

    def enqueue_linear_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_linear.Regenie2LinearChunkResult,
        host_dosage_buffer: HostGenotypeBuffer | None = None,
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
        chromosome_state = require_current_chromosome_state(
            self.current_chromosome_state,
            chromosome=self.current_chromosome,
        )

        genotype_device_array = put_genotype_matrix_on_device(
            genotype_matrix_by_variant,
            self.stage_timing_recorder,
            variant_metadata,
        )
        linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
        genotype_dosage_sum = put_chunk_array_on_device(
            linear_chunk_stats_arrays.dosage_sum,
            self.stage_timing_recorder,
            variant_metadata,
        )
        genotype_observation_count = put_chunk_array_on_device(
            linear_chunk_stats_arrays.observation_count,
            self.stage_timing_recorder,
            variant_metadata,
        )
        genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
            linear_chunk_stats_arrays.imputed_dosage_square_sum,
            self.stage_timing_recorder,
            variant_metadata,
        )
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state_variant_major(
            chromosome_state=chromosome_state,
            genotype_matrix_by_variant=genotype_device_array,
            genotype_dosage_sum=genotype_dosage_sum,
            genotype_observation_count=genotype_observation_count,
            genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
            score_dtype=self.score_dtype,
            linear_minimum_variance=self.linear_numerical_config.minimum_variance,
            linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
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
            self.score_dtype,
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
        chromosome_state = require_current_chromosome_state(
            self.current_chromosome_state,
            chromosome=self.current_chromosome,
        )

        genotype_device_array = put_genotype_matrix_on_device(
            genotype_matrix,
            self.stage_timing_recorder,
            variant_metadata,
        )
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_device_array,
            score_dtype=self.score_dtype,
            linear_minimum_variance=self.linear_numerical_config.minimum_variance,
            linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
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
        result_in_flight_limit: int | None = None,
        dosage_buffer_limit: int | None = None,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None = None,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
        output_statistic_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_sessions = writer_sessions
        self.committed_chunk_identifier_sets = committed_chunk_identifier_sets
        self.score_dtype = score_dtype
        self.linear_numerical_config = linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_matrix = put_compute_array_on_device(run_input.phenotype_matrix)
        self.regenie_state = regenie2_linear.prepare_regenie2_multi_linear_state(
            covariate_matrix=covariate_matrix,
            phenotype_matrix=phenotype_matrix,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_linear.Regenie2MultiLinearChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-multi-linear-callback",
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            output_statistic_dtype=output_statistic_dtype,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed multi-trait JAX results and write each trait in order."""
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
                multi_work_item = typing.cast("Regenie2MultiResultWriteWorkItem", work_item)
                self.process_multi_result_write_item(multi_work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def consume_result_write_items_without_timing(self) -> None:
        """Consume multi-trait result write items without diagnostic queue timing."""
        while True:
            work_item = self.result_queue.get()
            if work_item is None:
                return
            multi_work_item = typing.cast("Regenie2MultiResultWriteWorkItem", work_item)
            self.process_multi_result_write_item(multi_work_item)

    def process_multi_result_write_item(self, multi_work_item: Regenie2MultiResultWriteWorkItem) -> None:
        """Materialize and write one multi-trait linear result work item."""
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
                output_statistic_dtype=self.output_statistic_dtype,
            )
        finally:
            self.release_result_work_item_buffer(multi_work_item)

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
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state(
                chromosome_state=chromosome_state,
                genotype_matrix=genotype_device_array,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
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
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_observation_count = put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state_variant_major(
                chromosome_state=chromosome_state,
                genotype_matrix_by_variant=genotype_device_array,
                genotype_dosage_sum=genotype_dosage_sum,
                genotype_observation_count=genotype_observation_count,
                genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
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

    def compute_preprocessed_variant_major_packed8_chunk(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        packed_probability_pairs_by_variant: jax.Array | npt.NDArray[np.uint8],
        chunk_stats: _core.ChunkStats,
    ) -> None:
        """Compute one packed8 chunk and enqueue multi-trait results."""
        host_packed_buffer = self.get_releasable_dosage_buffer(packed_probability_pairs_by_variant)
        self.acquire_result_in_flight_slot()
        try:
            self.prepare_chromosome_state(variant_metadata)
            chromosome_state = require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            packed_device_array = put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
            )
            linear_chunk_stats_arrays = get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_observation_count = put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            genotype_imputed_dosage_square_sum = put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_multi_linear_chunk_packed8_donating_inputs(
                chromosome_state=chromosome_state,
                packed_probability_pairs_by_variant=packed_device_array,
                genotype_dosage_sum=genotype_dosage_sum,
                genotype_observation_count=genotype_observation_count,
                genotype_imputed_dosage_square_sum=genotype_imputed_dosage_square_sum,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_multi_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_packed_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_packed_buffer is not None:
                self.release_dosage_buffer(host_packed_buffer)
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
            self.score_dtype,
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
        host_dosage_buffer: HostGenotypeBuffer | None = None,
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


__all__ = [
    "LinearRegenie2PipelineCallback",
    "MultiLinearRegenie2PipelineCallback",
]
