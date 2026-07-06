"""Linear REGENIE step 2 callback implementations."""

from __future__ import annotations

import time
import typing

import jax

import g.engine.callbacks.runtime as runtime
import g.engine.callbacks.shared as shared
import g.engine.callbacks.transfers as transfers
import g.engine.callbacks.writers as writers
from g import _core, types
from g.compute.regenie2_linear import api as regenie2_linear
from g.compute.regenie2_linear import config as regenie2_linear_config
from g.engine import timing as engine_timing

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from g.runner import events


class LinearRegenie2PipelineCallback(runtime.NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for quantitative traits."""

    def __init__(
        self,
        run_input: shared.NativeBgenRunInputProtocol,
        prediction_source: shared.RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        staging_depth: int,
        native_callback_batch_size: int,
        result_in_flight_limit: int | None,
        dosage_buffer_limit: int | None,
        score_dtype: types.FloatingPointDtype,
        linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None,
        stage_timing_recorder: engine_timing.StageTimingRecorder | None,
        telemetry_session: events.TelemetrySession | None,
        output_statistic_dtype: types.FloatingPointDtype,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.score_dtype = score_dtype
        self.linear_numerical_config = linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG
        covariate_matrix = transfers.put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_vector = transfers.put_compute_array_on_device(run_input.phenotype_vector)
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
            native_callback_batch_size=native_callback_batch_size,
            expected_result_work_item_kind=runtime.ResultWriteItemKind.SINGLE_RESULT,
            flush_binary_correction_diagnostics_on_result_stop=False,
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
                shared.Regenie2ResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=True,
                    binary_chunk_diagnostics=None,
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
                shared.Regenie2ResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=True,
                    binary_chunk_diagnostics=None,
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
            chromosome_state = runtime.require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )

            packed_device_array = transfers.put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="packed_probability_pairs",
            )
            linear_chunk_stats_arrays = transfers.get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="dosage_sum",
            )
            genotype_observation_count = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="observation_count",
            )
            genotype_imputed_dosage_square_sum = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="imputed_dosage_square_sum",
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
            transfers.block_compute_result_for_timing(
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
        host_dosage_buffer: shared.HostGenotypeBuffer | None,
        release_in_flight_slot: bool,
    ) -> None:
        """Enqueue a linear result for materialization and writing."""
        self.put_result_write_item(
            shared.Regenie2ResultWriteWorkItem(
                metadata=variant_metadata,
                chunk_stats=chunk_stats,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=None,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=release_in_flight_slot,
                binary_chunk_diagnostics=None,
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
        chromosome_state = runtime.require_current_chromosome_state(
            self.current_chromosome_state,
            chromosome=self.current_chromosome,
        )

        genotype_device_array = transfers.put_genotype_matrix_on_device(
            genotype_matrix_by_variant,
            self.stage_timing_recorder,
            variant_metadata,
            array_role="genotype_matrix_by_variant",
        )
        linear_chunk_stats_arrays = transfers.get_linear_chunk_stats_arrays(chunk_stats)
        genotype_dosage_sum = transfers.put_chunk_array_on_device(
            linear_chunk_stats_arrays.dosage_sum,
            self.stage_timing_recorder,
            variant_metadata,
            array_role="dosage_sum",
        )
        genotype_observation_count = transfers.put_chunk_array_on_device(
            linear_chunk_stats_arrays.observation_count,
            self.stage_timing_recorder,
            variant_metadata,
            array_role="observation_count",
        )
        genotype_imputed_dosage_square_sum = transfers.put_chunk_array_on_device(
            linear_chunk_stats_arrays.imputed_dosage_square_sum,
            self.stage_timing_recorder,
            variant_metadata,
            array_role="imputed_dosage_square_sum",
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
        transfers.block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
        )
        return result

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached linear chromosome state for the metadata chromosome."""
        chromosome = str(variant_metadata.chromosome_label)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_predictions = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_linear.prepare_regenie2_linear_chromosome_state(
            self.regenie_state,
            loco_predictions,
            self.score_dtype,
        )
        jax.block_until_ready(self.current_chromosome_state.adjusted_residual)
        engine_timing.record_stage_duration(
            self.stage_timing_recorder,
            "chromosome_state_preparation",
            chromosome_start_time,
        )
        self.current_chromosome = chromosome

    def compute_linear_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
    ) -> regenie2_linear.Regenie2LinearChunkResult:
        """Compute quantitative REGENIE step 2 statistics for one chunk."""
        self.prepare_chromosome_state(variant_metadata)
        chromosome_state = runtime.require_current_chromosome_state(
            self.current_chromosome_state,
            chromosome=self.current_chromosome,
        )

        genotype_device_array = transfers.put_genotype_matrix_on_device(
            genotype_matrix,
            self.stage_timing_recorder,
            variant_metadata,
            array_role="genotype_matrix",
        )
        compute_start_time = time.perf_counter()
        result = regenie2_linear.compute_regenie2_linear_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_device_array,
            genotype_dosage_sum=None,
            genotype_observation_count=None,
            genotype_imputed_dosage_square_sum=None,
            score_dtype=self.score_dtype,
            linear_minimum_variance=self.linear_numerical_config.minimum_variance,
            linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
        )
        transfers.block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
        )
        return result


class MultiLinearRegenie2PipelineCallback(runtime.NativeBgenCallbackRunner):
    """Compute/write callback for quantitative multi-phenotype REGENIE step 2."""

    def __init__(
        self,
        run_input: shared.NativeBgenMultiRunInputProtocol,
        prediction_source: shared.MultiRegeniePredictionSourceProtocol,
        writer_sessions: tuple[typing.Any, ...],
        chunk_write_planner: _core.NativeMultiTraitChunkWritePlanner,
        staging_depth: int,
        native_callback_batch_size: int,
        result_in_flight_limit: int | None,
        dosage_buffer_limit: int | None,
        score_dtype: types.FloatingPointDtype,
        linear_numerical_config: regenie2_linear_config.LinearNumericalConfig | None,
        stage_timing_recorder: engine_timing.StageTimingRecorder | None,
        telemetry_session: events.TelemetrySession | None,
        output_statistic_dtype: types.FloatingPointDtype,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_sessions = writer_sessions
        self.chunk_write_planner = chunk_write_planner
        self.score_dtype = score_dtype
        self.linear_numerical_config = linear_numerical_config or regenie2_linear_config.DEFAULT_LINEAR_NUMERICAL_CONFIG
        covariate_matrix = transfers.put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_matrix = transfers.put_compute_array_on_device(run_input.phenotype_matrix)
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
            native_callback_batch_size=native_callback_batch_size,
            expected_result_work_item_kind=runtime.ResultWriteItemKind.MULTI_RESULT,
            flush_binary_correction_diagnostics_on_result_stop=False,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
            output_statistic_dtype=output_statistic_dtype,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed multi-trait JAX results and write each trait in order."""
        try:
            self.consume_multi_result_write_items_with_native_runtime_resources()
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

    def process_multi_result_write_item(self, multi_work_item: shared.Regenie2MultiResultWriteWorkItem) -> None:
        """Materialize and write one multi-trait linear result work item."""
        host_dosage_buffer_released = False
        try:
            materialized_chunk = writers.materialize_regenie2_multi_native_chunk_with_optional_timing(
                writer_sessions=self.writer_sessions,
                chunk_write_planner=self.chunk_write_planner,
                metadata=multi_work_item.metadata,
                beta=multi_work_item.beta,
                standard_error=multi_work_item.standard_error,
                chi_squared=multi_work_item.chi_squared,
                log10_p_value=multi_work_item.log10_p_value,
                extra_code=multi_work_item.extra_code,
                stage_timing_recorder=self.stage_timing_recorder,
                output_statistic_dtype=self.output_statistic_dtype,
            )
            host_dosage_buffer_released = self.release_result_work_item_host_buffer(multi_work_item)
            writers.write_materialized_regenie2_multi_native_chunk_with_optional_timing(
                metadata=multi_work_item.metadata,
                chunk_stats=multi_work_item.chunk_stats,
                materialized_chunk=materialized_chunk,
                stage_timing_recorder=self.stage_timing_recorder,
                output_statistic_dtype=self.output_statistic_dtype,
            )
        finally:
            self.release_result_work_item_final_resources(
                multi_work_item,
                host_dosage_buffer_released=host_dosage_buffer_released,
            )

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
            chromosome_state = runtime.require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = transfers.put_genotype_matrix_on_device(
                genotype_matrix,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="genotype_matrix",
            )
            compute_start_time = time.perf_counter()
            result = regenie2_linear.compute_regenie2_multi_linear_chunk_from_chromosome_state(
                chromosome_state=chromosome_state,
                genotype_matrix=genotype_device_array,
                genotype_dosage_sum=None,
                genotype_observation_count=None,
                genotype_imputed_dosage_square_sum=None,
                score_dtype=self.score_dtype,
                linear_minimum_variance=self.linear_numerical_config.minimum_variance,
                linear_relative_variance_tolerance=self.linear_numerical_config.relative_variance_tolerance,
            )
            transfers.block_compute_result_for_timing(
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
            chromosome_state = runtime.require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            genotype_device_array = transfers.put_genotype_matrix_on_device(
                genotype_matrix_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="genotype_matrix_by_variant",
            )
            linear_chunk_stats_arrays = transfers.get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="dosage_sum",
            )
            genotype_observation_count = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="observation_count",
            )
            genotype_imputed_dosage_square_sum = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="imputed_dosage_square_sum",
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
            transfers.block_compute_result_for_timing(
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
            chromosome_state = runtime.require_current_chromosome_state(
                self.current_chromosome_state,
                chromosome=self.current_chromosome,
            )
            packed_device_array = transfers.put_genotype_matrix_on_device(
                packed_probability_pairs_by_variant,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="packed_probability_pairs",
            )
            linear_chunk_stats_arrays = transfers.get_linear_chunk_stats_arrays(chunk_stats)
            genotype_dosage_sum = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="dosage_sum",
            )
            genotype_observation_count = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="observation_count",
            )
            genotype_imputed_dosage_square_sum = transfers.put_chunk_array_on_device(
                linear_chunk_stats_arrays.imputed_dosage_square_sum,
                self.stage_timing_recorder,
                variant_metadata,
                array_role="imputed_dosage_square_sum",
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
            transfers.block_compute_result_for_timing(
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
        chromosome = str(variant_metadata.chromosome_label)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_predictions = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_linear.prepare_regenie2_multi_linear_chromosome_state(
            self.regenie_state,
            loco_predictions,
            self.score_dtype,
        )
        jax.block_until_ready(self.current_chromosome_state.adjusted_residual_matrix)
        engine_timing.record_stage_duration(
            self.stage_timing_recorder,
            "chromosome_state_preparation",
            chromosome_start_time,
        )
        self.current_chromosome = chromosome

    def enqueue_multi_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_linear.Regenie2MultiLinearChunkResult,
        host_dosage_buffer: shared.HostGenotypeBuffer | None,
        release_in_flight_slot: bool,
    ) -> None:
        """Enqueue a multi-linear result for materialization and writing."""
        self.put_result_write_item(
            typing.cast(
                "shared.Regenie2ResultWriteWorkItem",
                shared.Regenie2MultiResultWriteWorkItem(
                    metadata=variant_metadata,
                    chunk_stats=chunk_stats,
                    beta=result.beta,
                    standard_error=result.standard_error,
                    chi_squared=result.chi_squared,
                    log10_p_value=result.log10_p_value,
                    extra_code=None,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=release_in_flight_slot,
                    binary_chunk_diagnostics=None,
                ),
            )
        )
