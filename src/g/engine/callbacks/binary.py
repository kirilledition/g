"""Binary and Firth REGENIE step 2 callback implementations."""

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
from g.compute.regenie2_binary import api as regenie2_binary
from g.compute.regenie2_binary import config as regenie2_binary_config
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
get_binary_chunk_stats_arrays = transfers.get_binary_chunk_stats_arrays
block_compute_result_for_timing = transfers.block_compute_result_for_timing
write_regenie2_multi_native_chunk_with_optional_timing = writers.write_regenie2_multi_native_chunk_with_optional_timing
block_until_ready = diagnostics.block_until_ready
enforce_null_logistic_nonconvergence_policy = diagnostics.enforce_null_logistic_nonconvergence_policy
collect_binary_chunk_diagnostics_if_needed = diagnostics.collect_binary_chunk_diagnostics_if_needed
get_metadata_chromosome = shared.get_metadata_chromosome


class BinaryRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback used by the native BGEN pipeline for binary traits."""

    def __init__(
        self,
        run_input: NativeBgenRunInputProtocol,
        prediction_source: RegeniePredictionSourceProtocol,
        writer_session: typing.Any,
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig,
        null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
            types.NullLogisticNonconvergencePolicy.FAIL
        ),
        staging_depth: int = 1,
        result_in_flight_limit: int | None = None,
        dosage_buffer_limit: int | None = None,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_session = writer_session
        self.correction_plan = correction_plan
        self.kernel_config = kernel_config
        self.null_logistic_nonconvergence_policy = null_logistic_nonconvergence_policy
        self.score_dtype = score_dtype
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_vector = put_compute_array_on_device(run_input.phenotype_vector)
        self.regenie_state = regenie2_binary.prepare_regenie2_binary_state(
            covariate_matrix=covariate_matrix,
            phenotype_vector=phenotype_vector,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_binary.Regenie2BinaryChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-binary-callback",
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
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
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            result = self.compute_binary_result(
                variant_metadata=variant_metadata,
                genotype_matrix=genotype_matrix,
                sparse_candidate_mask=sparse_candidate_mask,
            )
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=True,
                binary_chunk_diagnostics=collect_binary_chunk_diagnostics_if_needed(
                    stage_timing_recorder=self.stage_timing_recorder,
                    result=result,
                ),
            )
        except Exception:
            if host_dosage_buffer is not None:
                self.release_dosage_buffer(host_dosage_buffer)
            self.release_result_in_flight_slot()
            raise

    def enqueue_binary_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: (regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult),
        host_dosage_buffer: HostGenotypeBuffer | None = None,
        release_in_flight_slot: bool = False,
        binary_chunk_diagnostics: regenie2_binary.BinaryChunkDiagnostics | None = None,
    ) -> None:
        """Enqueue a binary result for materialization and writing."""
        self.put_result_write_item(
            Regenie2ResultWriteWorkItem(
                metadata=variant_metadata,
                chunk_stats=chunk_stats,
                beta=result.beta,
                standard_error=result.standard_error,
                chi_squared=result.chi_squared,
                log10_p_value=result.log10_p_value,
                extra_code=result.extra_code,
                binary_chunk_diagnostics=binary_chunk_diagnostics,
                host_dosage_buffer=host_dosage_buffer,
                release_in_flight_slot=release_in_flight_slot,
            )
        )

    def prepare_chromosome_state(self, variant_metadata: typing.Any) -> None:
        """Prepare cached binary chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_binary.prepare_regenie2_binary_chromosome_state(
            state=self.regenie_state,
            loco_offset=loco_offset,
            correction_plan=self.correction_plan,
            kernel_config=self.kernel_config,
            score_dtype=self.score_dtype,
        )
        chromosome_ready_value = getattr(
            self.current_chromosome_state,
            "score_residual",
            self.current_chromosome_state,
        )
        block_until_ready(chromosome_ready_value)
        enforce_null_logistic_nonconvergence_policy(
            chromosome=chromosome,
            null_logistic_converged=self.current_chromosome_state.null_logistic_converged,
            policy=self.null_logistic_nonconvergence_policy,
        )
        if self.stage_timing_recorder is not None:
            self.stage_timing_recorder.add_null_logistic_diagnostics(
                {
                    "chromosome": chromosome,
                    "iteration_count": int(jax.device_get(self.current_chromosome_state.null_logistic_iteration_count)),
                    "converged": int(jax.device_get(self.current_chromosome_state.null_logistic_converged)),
                    "firth_iteration_count": int(
                        jax.device_get(self.current_chromosome_state.null_firth_iteration_count)
                    ),
                    "firth_convergence_reason_code": int(
                        jax.device_get(self.current_chromosome_state.null_firth_convergence_reason_code)
                    ),
                    "correction_method": self.correction_plan.method.value,
                }
            )
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def compute_binary_result(
        self,
        *,
        variant_metadata: typing.Any,
        genotype_matrix: jax.Array | npt.NDArray[np.float32],
        sparse_candidate_mask: jax.Array | None = None,
    ) -> regenie2_binary.Regenie2BinaryScoreChunkResult | regenie2_binary.Regenie2BinaryChunkResult:
        """Compute binary REGENIE step 2 statistics for one chunk."""
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
        result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state(
            chromosome_state=chromosome_state,
            genotype_matrix=genotype_device_array,
            correction_plan=self.correction_plan,
            sparse_candidate_mask=sparse_candidate_mask,
            kernel_config=self.kernel_config,
            score_dtype=self.score_dtype,
            stage_duration_recorder=self.get_stage_duration_recorder(),
        )
        block_compute_result_for_timing(
            result_ready_value=result.log10_p_value,
            stage_timing_recorder=self.stage_timing_recorder,
            start_time=compute_start_time,
            chunk_metadata=variant_metadata,
        )
        return result

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
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_binary_score_test_variant_major_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_variant_major(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                binary_chunk_diagnostics=collect_binary_chunk_diagnostics_if_needed(
                    stage_timing_recorder=self.stage_timing_recorder,
                    result=result,
                ),
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
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_binary_score_test_packed8_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_binary_chunk_from_chromosome_state_packed8(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                )
            block_compute_result_for_timing(
                result_ready_value=result.log10_p_value,
                stage_timing_recorder=self.stage_timing_recorder,
                start_time=compute_start_time,
                chunk_metadata=variant_metadata,
            )
            self.enqueue_binary_result_for_write(
                variant_metadata=variant_metadata,
                chunk_stats=chunk_stats,
                result=result,
                binary_chunk_diagnostics=collect_binary_chunk_diagnostics_if_needed(
                    stage_timing_recorder=self.stage_timing_recorder,
                    result=result,
                ),
                host_dosage_buffer=host_packed_buffer,
                release_in_flight_slot=True,
            )
        except Exception:
            if host_packed_buffer is not None:
                self.release_dosage_buffer(host_packed_buffer)
            self.release_result_in_flight_slot()
            raise


class MultiBinaryRegenie2PipelineCallback(NativeBgenCallbackRunner):
    """Compute/write callback for binary multi-phenotype REGENIE step 2."""

    def __init__(
        self,
        run_input: NativeBgenMultiRunInputProtocol,
        prediction_source: MultiRegeniePredictionSourceProtocol,
        writer_sessions: tuple[typing.Any, ...],
        committed_chunk_identifier_sets: tuple[set[int], ...],
        correction_plan: types.BinaryCorrectionPlan,
        kernel_config: regenie2_binary_config.BinaryKernelConfig,
        null_logistic_nonconvergence_policy: types.NullLogisticNonconvergencePolicy = (
            types.NullLogisticNonconvergencePolicy.FAIL
        ),
        staging_depth: int = 1,
        result_in_flight_limit: int | None = None,
        dosage_buffer_limit: int | None = None,
        score_dtype: types.FloatingPointDtype = types.FloatingPointDtype.FLOAT32,
        stage_timing_recorder: timing.StageTimingRecorder | None = None,
        telemetry_session: telemetry.TelemetrySession | None = None,
    ) -> None:
        """Initialize the callback state."""
        self.run_input = run_input
        self.prediction_source = prediction_source
        self.writer_sessions = writer_sessions
        self.committed_chunk_identifier_sets = committed_chunk_identifier_sets
        self.correction_plan = correction_plan
        self.kernel_config = kernel_config
        self.null_logistic_nonconvergence_policy = null_logistic_nonconvergence_policy
        self.score_dtype = score_dtype
        covariate_matrix = put_compute_array_on_device(run_input.covariate_matrix)
        phenotype_matrix = put_compute_array_on_device(run_input.phenotype_matrix)
        self.regenie_state = regenie2_binary.prepare_regenie2_multi_binary_state(
            covariate_matrix=covariate_matrix,
            phenotype_matrix=phenotype_matrix,
            score_dtype=score_dtype,
        )
        self.current_chromosome: str | None = None
        self.current_chromosome_state: regenie2_binary.Regenie2MultiBinaryChromosomeState | None = None
        super().__init__(
            worker_name="regenie2-multi-binary-callback",
            staging_depth=staging_depth,
            result_in_flight_limit=result_in_flight_limit,
            dosage_buffer_limit=dosage_buffer_limit,
            stage_timing_recorder=stage_timing_recorder,
            telemetry_session=telemetry_session,
        )

    def consume_result_write_items(self) -> None:
        """Materialize computed multi-trait JAX results and write each trait in order."""
        try:
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
                    )
                finally:
                    self.release_result_work_item_buffer(multi_work_item)
        except Exception as error:  # noqa: BLE001
            self.result_worker_error = error

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
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state(
                chromosome_state=chromosome_state,
                genotype_matrix=genotype_device_array,
                correction_plan=self.correction_plan,
                sparse_candidate_mask=sparse_candidate_mask,
                kernel_config=self.kernel_config,
                score_dtype=self.score_dtype,
                stage_duration_recorder=self.get_stage_duration_recorder(),
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
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_multi_binary_score_test_variant_major_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_variant_major(
                    chromosome_state=chromosome_state,
                    genotype_matrix_by_variant=genotype_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
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
        """Compute one packed8 chunk and enqueue multi-trait binary results."""
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
            binary_chunk_stats_arrays = get_binary_chunk_stats_arrays(
                chunk_stats,
                include_sparse_firth_candidate=self.correction_plan.method != types.BinaryFallbackMethod.SCORE_ONLY,
            )
            dosage_sum = put_chunk_array_on_device(
                binary_chunk_stats_arrays.dosage_sum,
                self.stage_timing_recorder,
                variant_metadata,
            )
            observation_count = put_chunk_array_on_device(
                binary_chunk_stats_arrays.observation_count,
                self.stage_timing_recorder,
                variant_metadata,
            )
            sparse_candidate_mask = (
                None
                if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY
                else put_chunk_array_on_device(
                    typing.cast("npt.NDArray[np.bool_]", binary_chunk_stats_arrays.sparse_candidate_mask),
                    self.stage_timing_recorder,
                    variant_metadata,
                )
            )
            compute_start_time = time.perf_counter()
            if self.correction_plan.method == types.BinaryFallbackMethod.SCORE_ONLY:
                compute_score_test = regenie2_binary.compute_multi_binary_score_test_packed8_donating_inputs
                result = compute_score_test(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    kernel_config=self.kernel_config,
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
                    score_dtype=self.score_dtype,
                )
            else:
                result = regenie2_binary.compute_regenie2_multi_binary_chunk_from_chromosome_state_packed8(
                    chromosome_state=chromosome_state,
                    packed_probability_pairs_by_variant=packed_device_array,
                    correction_plan=self.correction_plan,
                    sparse_candidate_mask=sparse_candidate_mask,
                    kernel_config=self.kernel_config,
                    score_dtype=self.score_dtype,
                    stage_duration_recorder=self.get_stage_duration_recorder(),
                    dosage_sum=dosage_sum,
                    observation_count=observation_count,
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
        """Prepare cached multi-binary chromosome state for the metadata chromosome."""
        chromosome = get_metadata_chromosome(variant_metadata)
        if chromosome == self.current_chromosome:
            return
        chromosome_start_time = time.perf_counter()
        loco_offset = jax.device_put(self.prediction_source.get_chromosome_predictions(chromosome))
        self.current_chromosome_state = regenie2_binary.prepare_regenie2_multi_binary_chromosome_state(
            self.regenie_state,
            loco_offset,
            self.correction_plan,
            self.kernel_config,
            self.score_dtype,
        )
        block_until_ready(self.current_chromosome_state.score_residual)
        enforce_null_logistic_nonconvergence_policy(
            chromosome=chromosome,
            null_logistic_converged=self.current_chromosome_state.null_logistic_converged,
            policy=self.null_logistic_nonconvergence_policy,
            phenotype_names=self.run_input.phenotype_names,
        )
        if self.stage_timing_recorder is not None:
            iteration_counts = jax.device_get(self.current_chromosome_state.null_logistic_iteration_count)
            convergence_flags = jax.device_get(self.current_chromosome_state.null_logistic_converged)
            for trait_index, phenotype_name in enumerate(self.run_input.phenotype_names):
                self.stage_timing_recorder.add_null_logistic_diagnostics(
                    {
                        "chromosome": chromosome,
                        "phenotype": phenotype_name,
                        "iteration_count": int(iteration_counts[trait_index]),
                        "converged": int(convergence_flags[trait_index]),
                        "correction_method": self.correction_plan.method.value,
                    }
                )
        timing.record_stage_duration(self.stage_timing_recorder, "chromosome_state_preparation", chromosome_start_time)
        self.current_chromosome = chromosome

    def enqueue_multi_result_for_write(
        self,
        *,
        variant_metadata: _core.VariantMetadata,
        chunk_stats: _core.ChunkStats,
        result: regenie2_binary.Regenie2MultiBinaryScoreChunkResult | regenie2_binary.Regenie2MultiBinaryChunkResult,
        host_dosage_buffer: HostGenotypeBuffer | None = None,
        release_in_flight_slot: bool = False,
    ) -> None:
        """Enqueue a multi-binary result for materialization and writing."""
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
                    extra_code=result.extra_code,
                    host_dosage_buffer=host_dosage_buffer,
                    release_in_flight_slot=release_in_flight_slot,
                ),
            )
        )


__all__ = [
    "BinaryRegenie2PipelineCallback",
    "MultiBinaryRegenie2PipelineCallback",
]
