from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt

import g.types

class InputConfig:
    @property
    def bgen(self) -> Path | None: ...
    @property
    def sample(self) -> Path | None: ...
    @property
    def pheno_file(self) -> Path | None: ...
    @property
    def pheno_columns(self) -> tuple[str, ...]: ...
    @property
    def covar_file(self) -> Path | None: ...
    @property
    def covar_columns(self) -> tuple[str, ...]: ...
    @property
    def pred(self) -> Path | None: ...

class TraitConfig:
    @property
    def step(self) -> int: ...
    @property
    def trait_type(self) -> g.types.RegenieTraitType: ...
    @property
    def bsize(self) -> int: ...
    @property
    def threads(self) -> int | None: ...

class BinaryConfig:
    @property
    def firth(self) -> bool: ...
    @property
    def approx(self) -> bool: ...
    @property
    def spa(self) -> bool: ...
    @property
    def p_threshold(self) -> float: ...
    @property
    def firth_se(self) -> bool: ...

class GComputeConfig:
    @property
    def device(self) -> g.types.Device: ...
    @property
    def staging_depth(self) -> int: ...
    @property
    def native_callback_batch_size(self) -> int: ...
    @property
    def result_in_flight_limit(self) -> int | None: ...
    @property
    def dosage_buffer_limit(self) -> int | None: ...
    @property
    def variant_limit(self) -> int | None: ...
    @property
    def trusted_no_missing_diploid(self) -> bool: ...
    @property
    def trusted_bgen_validation_mode(self) -> g.types.TrustedBgenValidationMode: ...
    @property
    def sample_key_mode(self) -> g.types.SampleKeyMode: ...
    @property
    def multi_phenotype_sample_mode(self) -> g.types.MultiPhenotypeSampleMode: ...
    @property
    def firth_batch_size(self) -> int: ...
    @property
    def firth_candidate_capacity(self) -> int: ...
    @property
    def binary_null_maximum_iterations(self) -> int: ...
    @property
    def binary_null_coefficient_tolerance(self) -> float: ...
    @property
    def null_logistic_nonconvergence_policy(self) -> g.types.NullLogisticNonconvergencePolicy: ...
    @property
    def binary_minimum_probability(self) -> float: ...
    @property
    def binary_minimum_variance(self) -> float: ...
    @property
    def binary_relative_variance_tolerance(self) -> float: ...
    @property
    def linear_minimum_variance(self) -> float: ...
    @property
    def linear_relative_variance_tolerance(self) -> float: ...
    @property
    def firth_maximum_iterations(self) -> int: ...
    @property
    def firth_gradient_tolerance(self) -> float: ...
    @property
    def firth_coefficient_tolerance(self) -> float: ...
    @property
    def firth_likelihood_tolerance(self) -> float: ...
    @property
    def firth_maximum_step_size(self) -> float: ...
    @property
    def firth_pseudo_maximum_iterations(self) -> int: ...
    @property
    def firth_pseudo_inner_maximum_iterations(self) -> int: ...
    @property
    def firth_newton_raphson_zero_start_iterations(self) -> int: ...
    @property
    def firth_line_search_maximum_attempts(self) -> int: ...
    @property
    def firth_step_halving_maximum_attempts(self) -> int: ...
    @property
    def firth_initial_response_scale(self) -> float: ...
    @property
    def firth_sparse_carrier_dosage_threshold(self) -> float: ...
    @property
    def firth_step_halving_scale(self) -> float: ...
    @property
    def null_firth_maximum_iterations(self) -> int: ...
    @property
    def null_firth_gradient_tolerance(self) -> float: ...
    @property
    def null_firth_maximum_step_size(self) -> float: ...
    @property
    def null_firth_fallback_iteration_multiplier(self) -> int: ...
    @property
    def null_firth_fallback_step_divisor(self) -> float: ...
    @property
    def null_firth_line_search_maximum_attempts(self) -> int: ...
    @property
    def null_firth_step_halving_scale(self) -> float: ...
    @property
    def use_block_firth_math(self) -> bool: ...
    @property
    def bgen_decode_tile_variant_count(self) -> int: ...
    @property
    def gpu_genotype_format(self) -> g.types.GpuGenotypeFormat: ...
    @property
    def score_dtype(self) -> g.types.FloatingPointDtype: ...
    @property
    def firth_dtype(self) -> g.types.FloatingPointDtype: ...
    @property
    def jax_cache_dir(self) -> Path | None: ...
    @property
    def jax_matmul_precision(self) -> g.types.JaxMatmulPrecision | None: ...
    @property
    def jax_persistent_cache(self) -> bool: ...
    @property
    def jax_persistent_cache_min_entry_size_bytes(self) -> int: ...
    @property
    def jax_persistent_cache_min_compile_time_seconds(self) -> int: ...
    @property
    def jax_xla_autotune_cache(self) -> bool: ...
    @property
    def jax_transfer_guard(self) -> bool: ...

class GOutputConfig:
    @property
    def out(self) -> Path | None: ...
    @property
    def format(self) -> g.types.OutputFormat: ...
    @property
    def output_run_directory(self) -> Path | None: ...
    @property
    def writer_threads(self) -> int: ...
    @property
    def writer_queue_depth(self) -> int: ...
    @property
    def chunks_per_arrow_file(self) -> int: ...
    @property
    def arrow_compression(self) -> g.types.ArrowCompression: ...
    @property
    def parquet_compression(self) -> g.types.ParquetCompression: ...
    @property
    def output_statistic_dtype(self) -> g.types.FloatingPointDtype: ...
    @property
    def resume(self) -> bool: ...
    @property
    def resume_mode(self) -> g.types.ResumeMode: ...
    @property
    def finalize_parquet(self) -> bool: ...

class GDiagnosticsConfig:
    @property
    def telemetry(self) -> g.types.TelemetryMode: ...
    @property
    def log_dir(self) -> Path | None: ...
    @property
    def stage_timings_json(self) -> Path | None: ...
    @property
    def log_filter(self) -> str: ...
    @property
    def log_file(self) -> Path | None: ...
    @property
    def log_stderr(self) -> bool: ...
    @property
    def progress_interval_seconds(self) -> float: ...
    @property
    def progress_interval_chunks(self) -> int: ...
    @property
    def profile_summary_json(self) -> Path | None: ...
    @property
    def trace_file(self) -> Path | None: ...
    @property
    def trace_filter(self) -> str: ...
    @property
    def trace_event_cap(self) -> int: ...
    @property
    def log_queue_size(self) -> int: ...
    @property
    def log_lossy(self) -> bool: ...
    @property
    def include_source_location(self) -> bool: ...
    @property
    def include_span_events(self) -> bool: ...

class RegenieConfig:
    @staticmethod
    def from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig: ...
    @staticmethod
    def from_toml(path: str | Path) -> RegenieConfig: ...
    @property
    def input(self) -> InputConfig: ...
    @property
    def trait_(self) -> TraitConfig: ...
    @property
    def trait(self) -> TraitConfig: ...
    @property
    def binary(self) -> BinaryConfig: ...
    @property
    def g_compute(self) -> GComputeConfig: ...
    @property
    def g_output(self) -> GOutputConfig: ...
    @property
    def g_diagnostics(self) -> GDiagnosticsConfig: ...
    @property
    def is_validated(self) -> bool: ...
    def to_toml(self) -> str: ...

class CliOutcome:
    @property
    def exit_code(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...
    @property
    def config(self) -> RegenieConfig | None: ...

class ChunkStatsComputeArrays(typing.TypedDict, total=False):
    dosage_sum: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    imputed_dosage_square_sum: npt.NDArray[np.float32]
    is_rare_sparse_firth_candidate: npt.NDArray[np.bool_]

class ChunkSpec:
    variant_start_index: int
    variant_stop_index: int

class ChunkStats:
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    dosage_sum: npt.NDArray[np.float32]
    allele_count: npt.NDArray[np.float32]
    has_missing_values: bool
    dosage_square_sum: npt.NDArray[np.float32]
    imputed_dosage_square_sum: npt.NDArray[np.float32]
    info_score: npt.NDArray[np.float32]
    minor_allele_count: npt.NDArray[np.float32]
    zero_count: npt.NDArray[np.int32]
    nonzero_count: npt.NDArray[np.int32]
    is_sparse_candidate: npt.NDArray[np.bool_]
    is_rare_sparse_firth_candidate: npt.NDArray[np.bool_]
    def compute_arrays(
        self,
        *,
        include_imputed_dosage_square_sum: bool = True,
        include_sparse_firth_candidate: bool = True,
    ) -> ChunkStatsComputeArrays: ...

class VariantMetadata:
    variant_start_index: int
    variant_stop_index: int
    chromosome_label: str
    chromosome: list[str]
    variant_identifiers: list[str]
    position: npt.NDArray[np.int64]
    allele_one: list[str]
    allele_two: list[str]

class NativeAlignedSampleData:
    sample_indices: npt.NDArray[np.int64]
    family_identifiers: list[str]
    individual_identifiers: list[str]
    phenotype_name: str
    phenotype_vector: npt.NDArray[np.float32]
    covariate_names: list[str]
    covariate_matrix: npt.NDArray[np.float32]
    is_binary_trait: bool

class NativeMultiAlignedSampleData:
    sample_indices: npt.NDArray[np.int64]
    family_identifiers: list[str]
    individual_identifiers: list[str]
    phenotype_names: list[str]
    phenotype_matrix: npt.NDArray[np.float32]
    covariate_names: list[str]
    covariate_matrix: npt.NDArray[np.float32]
    is_binary_trait: bool

class NativeAlignedPhenotypeGroup:
    phenotype_indices: list[int]
    aligned_sample_data: NativeMultiAlignedSampleData

class NativeGroupedAlignedSampleData:
    groups: list[NativeAlignedPhenotypeGroup]

class NativeResolvedPhenotypeComputeGroup:
    group_mode: str
    phenotype_indices: list[int]
    phenotype_names: list[str]
    sample_mode: str
    sample_set_fingerprint: str
    covariate_design_fingerprint: str
    prediction_alignment_fingerprint: str | None

class Regenie2RunEngine:
    sample_count: int
    variant_count: int
    contains_embedded_samples: bool

    def __init__(
        self,
        bgen_path: str,
        chunk_size: int,
        variant_limit: int | None = None,
        trusted_no_missing_diploid: bool = False,
    ) -> None: ...
    def sample_identifiers(self) -> list[str]: ...
    def align_sample_data(
        self,
        sample_path: str | None,
        phenotype_path: str,
        phenotype_name: str,
        covariate_path: str | None = None,
        covariate_names: list[str] | None = None,
        is_binary_trait: bool = False,
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> NativeAlignedSampleData: ...
    def align_multi_sample_data(
        self,
        sample_path: str | None,
        phenotype_path: str,
        phenotype_names: list[str],
        covariate_path: str | None = None,
        covariate_names: list[str] | None = None,
        is_binary_trait: bool = False,
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> NativeMultiAlignedSampleData: ...
    def align_grouped_sample_data(
        self,
        sample_path: str | None,
        phenotype_path: str,
        phenotype_names: list[str],
        covariate_path: str | None = None,
        covariate_names: list[str] | None = None,
        is_binary_trait: bool = False,
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> NativeGroupedAlignedSampleData: ...
    def chromosome_boundary_indices(self) -> list[int]: ...
    def required_chromosomes(self, variant_limit: int | None = None) -> list[str]: ...
    def reset_profile(self) -> None: ...
    def profile_snapshot(self) -> dict[str, int]: ...
    def validate_trusted_no_missing_diploid(self) -> None: ...
    def mark_trusted_no_missing_diploid_validated(self) -> None: ...
    def variant_metadata_slice(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str]]: ...
    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: npt.NDArray[np.int64],
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        callback_batch_size: int = 1,
    ) -> int: ...
    def run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples(
        self,
        aligned_sample_data: NativeAlignedSampleData,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        callback_batch_size: int = 1,
    ) -> int: ...
    def run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples(
        self,
        aligned_sample_data: NativeMultiAlignedSampleData,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        callback_batch_size: int = 1,
    ) -> int: ...
    def run_bgen_variant_major_packed8_probability_pair_buffered_chunks(
        self,
        sample_indices: npt.NDArray[np.int64],
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int: ...
    def run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_aligned_samples(
        self,
        aligned_sample_data: NativeAlignedSampleData,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int: ...
    def run_bgen_variant_major_packed8_probability_pair_buffered_chunks_for_native_multi_aligned_samples(
        self,
        aligned_sample_data: NativeMultiAlignedSampleData,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int: ...

class RegeniePredictionSource:
    def __init__(
        self,
        prediction_list_path: str,
        phenotype_name: str,
        sample_family_identifiers: list[str],
        sample_individual_identifiers: list[str],
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> None: ...
    @staticmethod
    def from_native_aligned_sample_data(
        prediction_list_path: str,
        phenotype_name: str,
        aligned_sample_data: NativeAlignedSampleData,
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> RegeniePredictionSource: ...
    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]: ...

class MultiRegeniePredictionSource:
    def __init__(
        self,
        prediction_list_path: str,
        phenotype_names: list[str],
        sample_family_identifiers: list[str],
        sample_individual_identifiers: list[str],
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> None: ...
    @staticmethod
    def from_native_multi_aligned_sample_data(
        prediction_list_path: str,
        aligned_sample_data: NativeMultiAlignedSampleData,
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> MultiRegeniePredictionSource: ...
    @staticmethod
    def from_native_grouped_aligned_sample_data(
        prediction_list_path: str,
        grouped_aligned_sample_data: NativeGroupedAlignedSampleData,
        sample_key_mode: g.types.SampleKeyMode | str = "iid",
    ) -> list[MultiRegeniePredictionSource]: ...
    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]: ...

class NativeTelemetryProgressThrottle:
    def __init__(self, progress_interval_seconds: float, progress_interval_chunks: int) -> None: ...
    def should_emit_progress(self, processed_chunk_count: int) -> bool: ...

class NativeTelemetrySession:
    def __init__(
        self,
        stream_file: str,
        queue_size: int = 65536,
        lossy: bool = True,
        event_cap: int | None = None,
    ) -> None: ...
    def emit_json_line(self, json_line: str) -> None: ...
    def emit_payload(self, payload: dict[str, object]) -> None: ...
    def emit_event(
        self,
        run_id: str,
        event: str,
        level: str,
        timestamp: str,
        process_identifier: int,
        thread_name: str,
        fields: dict[str, object],
    ) -> None: ...
    def emit_current_event(
        self,
        run_id: str,
        event: str,
        level: str,
        fields: dict[str, object],
    ) -> None: ...
    def build_event_payload(
        self,
        run_id: str,
        event: str,
        level: str,
        timestamp: str,
        process_identifier: int,
        thread_name: str,
        fields: dict[str, object],
    ) -> dict[str, object]: ...
    def counters(self) -> dict[str, object]: ...
    def finish(self) -> dict[str, object]: ...
    def finish_with_close_event(
        self,
        run_id: str,
        timestamp: str,
        process_identifier: int,
        thread_name: str,
    ) -> dict[str, object]: ...
    def finish_with_current_close_event(self, run_id: str) -> dict[str, object]: ...

class NativeBinaryCorrectionSummary:
    @property
    def chunk_count(self) -> int: ...
    @property
    def score_only_count(self) -> int: ...
    @property
    def score_test_candidate_count(self) -> int: ...
    @property
    def firth_attempted_count(self) -> int: ...
    @property
    def firth_success_count(self) -> int: ...
    @property
    def firth_failed_count(self) -> int: ...
    @property
    def firth_numerical_failure_count(self) -> int: ...
    @property
    def firth_max_iteration_failure_count(self) -> int: ...
    @property
    def firth_invalid_statistic_failure_count(self) -> int: ...
    @property
    def firth_step_halving_failure_count(self) -> int: ...
    @property
    def pseudo_firth_attempt_count(self) -> int: ...
    @property
    def pseudo_firth_success_count(self) -> int: ...
    @property
    def nr_zero_start_attempt_count(self) -> int: ...
    @property
    def nr_zero_start_success_count(self) -> int: ...
    @property
    def nr_warm_start_attempt_count(self) -> int: ...
    @property
    def nr_warm_start_success_count(self) -> int: ...
    @property
    def sparse_correction_count(self) -> int: ...
    @property
    def dense_correction_count(self) -> int: ...
    @property
    def null_model_failure_count(self) -> int: ...
    def __init__(self) -> None: ...
    def add_null_model_failure_count(self, failure_count: int) -> None: ...
    def add_diagnostics_mapping(self, diagnostics: dict[str, int | float]) -> None: ...
    def add_diagnostics_counts(
        self,
        score_only_count: int,
        score_test_candidate_count: int,
        firth_candidate_count: int,
        firth_converged_count: int,
        firth_failed_count: int,
        firth_numerical_failure_count: int,
        firth_max_iteration_failure_count: int,
        firth_invalid_statistic_failure_count: int,
        firth_step_halving_failure_count: int,
        pseudo_firth_attempt_count: int,
        pseudo_firth_success_count: int,
        nr_zero_start_attempt_count: int,
        nr_zero_start_success_count: int,
        nr_warm_start_attempt_count: int,
        nr_warm_start_success_count: int,
        sparse_correction_count: int,
        dense_correction_count: int,
    ) -> None: ...
    def add_diagnostics_totals(
        self,
        chunk_count: int,
        score_only_count: int,
        score_test_candidate_count: int,
        firth_candidate_count: int,
        firth_converged_count: int,
        firth_failed_count: int,
        firth_numerical_failure_count: int,
        firth_max_iteration_failure_count: int,
        firth_invalid_statistic_failure_count: int,
        firth_step_halving_failure_count: int,
        pseudo_firth_attempt_count: int,
        pseudo_firth_success_count: int,
        nr_zero_start_attempt_count: int,
        nr_zero_start_success_count: int,
        nr_warm_start_attempt_count: int,
        nr_warm_start_success_count: int,
        sparse_correction_count: int,
        dense_correction_count: int,
    ) -> None: ...
    def should_emit(self) -> bool: ...
    def summary_payload(self) -> dict[str, int]: ...

class NativeCallbackQueueLimits:
    @property
    def dosage_queue_depth(self) -> int: ...
    @property
    def result_queue_depth(self) -> int: ...
    @property
    def result_in_flight_limit(self) -> int: ...
    @property
    def dosage_buffer_limit(self) -> int: ...

class NativeCallbackQueueOperationObservationPlan:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def blocked_seconds(self) -> float: ...

class NativeCallbackQueueStageObservationPlan:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def stage_name(self) -> str: ...
    @property
    def blocked_seconds(self) -> float: ...

class NativeCallbackChunkIdentity:
    @property
    def chunk_identifier(self) -> int: ...
    @property
    def chromosome(self) -> str: ...
    @property
    def variant_start_index(self) -> int: ...
    @property
    def variant_stop_index(self) -> int: ...
    @property
    def variant_count(self) -> int: ...

class NativeCallbackProgressUpdate:
    @property
    def processed_chunk_count(self) -> int: ...
    @property
    def completed_chromosome(self) -> str | None: ...
    @property
    def completed_processed_chunk_count(self) -> int | None: ...
    @property
    def started_chromosome(self) -> str | None: ...
    @property
    def chunk_identity(self) -> NativeCallbackChunkIdentity: ...

class NativeCallbackProgressCompletion:
    @property
    def chromosome(self) -> str: ...
    @property
    def processed_chunk_count(self) -> int: ...

class NativeCallbackProgressState:
    def __init__(self) -> None: ...
    @property
    def processed_chunk_count(self) -> int: ...
    @property
    def current_progress_chromosome(self) -> str | None: ...
    def record_processed_chunk(self, chunk_identity: NativeCallbackChunkIdentity) -> NativeCallbackProgressUpdate: ...
    def record_processed_chunk_without_progress(self) -> None: ...
    def finish_progress(self) -> NativeCallbackProgressCompletion | None: ...

class NativeDosageBufferReusePlan:
    @property
    def requires_slice(self) -> bool: ...
    @property
    def slice_dimensions(self) -> list[int]: ...

class NativeVariantMajorDosageBatchHandoffPlan:
    @property
    def chunk_count(self) -> int: ...

class NativeMultiTraitChunkWritePlan:
    @property
    def active_trait_indices(self) -> list[int]: ...
    @property
    def total_trait_count(self) -> int: ...
    @property
    def active_trait_count(self) -> int: ...
    @property
    def all_traits_committed(self) -> bool: ...

class NativeWriterFinishExecutionPlan:
    @property
    def writer_session_count(self) -> int: ...
    @property
    def thread_count(self) -> int: ...
    @property
    def has_writer_sessions(self) -> bool: ...
    @property
    def uses_parallel_finish(self) -> bool: ...

class NativeSingleTraitOutputWritePlan:
    @property
    def method_name(self) -> str: ...
    @property
    def uses_float64_native_writer(self) -> bool: ...

class NativeMultiTraitOutputWritePlan:
    @property
    def active_trait_count(self) -> int: ...
    @property
    def use_native_multi_writer(self) -> bool: ...
    @property
    def uses_float64_native_writer(self) -> bool: ...

class NativeCallbackWorkerLifecycleState:
    def __init__(self) -> None: ...
    @property
    def has_started(self) -> bool: ...
    def mark_started(self) -> bool: ...

class NativeCallbackWorkerShutdownTimeouts:
    @property
    def dosage_worker_join_timeout_seconds(self) -> float: ...
    @property
    def result_worker_join_timeout_seconds(self) -> float: ...
    @property
    def graceful_dosage_worker_join_timeout_seconds(self) -> float: ...
    @property
    def graceful_result_worker_join_timeout_seconds(self) -> float: ...
    @property
    def worker_abort_stop_timeout_seconds(self) -> float: ...

class NativeCallbackWorkerJoinPlan:
    @property
    def should_join(self) -> bool: ...
    @property
    def timeout_seconds(self) -> float: ...

class NativeCallbackWorkerStopPlan:
    @property
    def should_stop(self) -> bool: ...
    @property
    def timeout_seconds(self) -> float: ...

class NativeCallbackWorkerFinishPlan:
    @property
    def dosage_stop_timeout_seconds(self) -> float: ...
    @property
    def dosage_join_timeout_seconds(self) -> float: ...
    @property
    def result_stop_timeout_seconds(self) -> float: ...
    @property
    def result_join_timeout_seconds(self) -> float: ...

class NativeCallbackWorkerAbortPlan:
    @property
    def dosage_stop_timeout_seconds(self) -> float: ...
    @property
    def result_stop_timeout_seconds(self) -> float: ...

class NativeCallbackWorkerStopPollPlan:
    @property
    def should_stop(self) -> bool: ...
    @property
    def poll_timeout_seconds(self) -> float: ...

class NativeDosageBufferPoolState:
    def __init__(self, buffer_limit: int) -> None: ...
    @property
    def buffer_limit(self) -> int: ...
    @property
    def allocated_count(self) -> int: ...
    @property
    def buffer_identifiers(self) -> list[int]: ...
    def has_available_slot(self) -> bool: ...
    def owns_buffer(self, buffer_identifier: int) -> bool: ...
    def register_buffer(self, buffer_identifier: int) -> bool: ...
    def discard_buffer(self, buffer_identifier: int) -> bool: ...

class NativeResultInFlightSlotState:
    def __init__(self, slot_limit: int) -> None: ...
    @property
    def slot_limit(self) -> int: ...
    @property
    def occupied_count(self) -> int: ...
    def has_available_slot(self) -> bool: ...
    def acquire_slot(self) -> bool: ...
    def release_slot(self) -> bool: ...

class NativeStageTimingRecorder:
    exact_stage_timings: bool
    def __init__(self, exact_stage_timings: bool) -> None: ...
    def should_collect_exact_stage_timings(self) -> bool: ...
    def add_stage_duration(self, stage_name: str, duration_seconds: float) -> None: ...
    def add_chunk_stage_duration(
        self,
        chunk_identifier: int,
        chromosome: str,
        variant_start_index: int,
        variant_stop_index: int,
        variant_count: int,
        stage_name: str,
        duration_seconds: float,
    ) -> None: ...
    def set_native_bgen_profile(self, profile_snapshot: dict[str, int]) -> None: ...
    def add_binary_chunk_diagnostics(self, diagnostics: dict[str, int | float]) -> None: ...
    def add_null_logistic_diagnostics(self, diagnostics: dict[str, int | str]) -> None: ...
    def add_queue_backpressure_observation(
        self,
        queue_name: str,
        operation_name: str,
        queue_depth: int,
        queue_capacity: int,
        elapsed_seconds: float,
        blocked_seconds: float,
    ) -> None: ...
    def add_transfer_metadata(
        self,
        transfer_name: str,
        array_role: str,
        dtype_name: str,
        ndim: int,
        byte_count: int,
        element_count: int,
    ) -> None: ...
    def snapshot_payload(self) -> dict[str, object]: ...
    def stage_timing_json_payload(self) -> dict[str, object]: ...
    def write_stage_timing_snapshot(self, path: str) -> None: ...
    def derived_metrics_payload(self) -> dict[str, float]: ...
    def profile_summary_payload(self, run_id: str | None) -> dict[str, object]: ...
    def write_profile_summary(self, path: str, run_id: str | None) -> None: ...

class NativeRuntimeState:
    rayon_thread_count: int | None
    def __init__(self) -> None: ...
    def logging_runtime_policy_payload(self) -> dict[str, object] | None: ...
    def jax_runtime_policy_payload(self) -> dict[str, object] | None: ...
    def require_compatible_logging_runtime_policy(self, payload: dict[str, object]) -> None: ...
    def record_logging_runtime_policy(self, payload: dict[str, object]) -> None: ...
    def require_compatible_rayon_thread_count(self, thread_count: int | None) -> None: ...
    def record_rayon_thread_count(self, thread_count: int) -> None: ...
    def effective_rayon_thread_count(self, requested_thread_count: int | None) -> int | None: ...
    def require_compatible_jax_runtime_policy(self, payload: dict[str, object]) -> None: ...
    def record_jax_runtime_policy(self, payload: dict[str, object]) -> None: ...

class NativeShutdownController:
    def __init__(self) -> None: ...
    def reset(self) -> None: ...
    def requested_signal_payload(self) -> dict[str, object] | None: ...
    def request_shutdown_payload(self, signal_number: int) -> dict[str, object]: ...

class OutputWriterSession:
    def __init__(
        self,
        run_directory: str,
        chunks_directory: str,
        association_mode: g.types.AssociationMode | str,
        writer_thread_count: int,
        writer_queue_depth: int,
        output_format: g.types.OutputFormat | str,
        output_statistic_dtype: g.types.FloatingPointDtype | str,
        finalize_parquet: bool,
        chunks_per_arrow_file: int,
        arrow_compression: g.types.ArrowCompression | str,
        parquet_compression: g.types.ParquetCompression | str,
        collect_stage_timings: bool,
    ) -> None: ...
    def write_regenie2_native_chunk(
        self,
        *,
        metadata: VariantMetadata,
        chunk_stats: ChunkStats,
        beta: npt.NDArray[np.float32],
        standard_error: npt.NDArray[np.float32],
        chi_squared: npt.NDArray[np.float32],
        log10_p_value: npt.NDArray[np.float32],
        extra_code: npt.NDArray[np.int32] | None = None,
    ) -> None: ...
    def write_regenie2_native_chunk_f64(
        self,
        *,
        metadata: VariantMetadata,
        chunk_stats: ChunkStats,
        beta: npt.NDArray[np.float64],
        standard_error: npt.NDArray[np.float64],
        chi_squared: npt.NDArray[np.float64],
        log10_p_value: npt.NDArray[np.float64],
        extra_code: npt.NDArray[np.int32] | None = None,
    ) -> None: ...
    def finish(self) -> str | None: ...
    def finish_interrupted(self, signal_name: str) -> None: ...
    def abort(self) -> None: ...

class NativeOutputRunPaths:
    @property
    def run_directory(self) -> str: ...
    @property
    def chunks_directory(self) -> str: ...

class NativePreparedOutputRun:
    @property
    def run_directory(self) -> str: ...
    @property
    def chunks_directory(self) -> str: ...
    @property
    def existing_manifest_json(self) -> str | None: ...

class NativeInitializedOutputRun:
    @property
    def committed_chunk_identifiers(self) -> list[int]: ...

def write_regenie2_multi_native_chunk(
    *,
    writer_sessions: list[OutputWriterSession],
    active_trait_indices: list[int],
    metadata: VariantMetadata,
    chunk_stats: ChunkStats,
    beta: npt.NDArray[np.float32],
    standard_error: npt.NDArray[np.float32],
    chi_squared: npt.NDArray[np.float32],
    log10_p_value: npt.NDArray[np.float32],
    extra_code: npt.NDArray[np.int32] | None = None,
) -> None: ...
def write_regenie2_multi_native_chunk_f64(
    *,
    writer_sessions: list[OutputWriterSession],
    active_trait_indices: list[int],
    metadata: VariantMetadata,
    chunk_stats: ChunkStats,
    beta: npt.NDArray[np.float64],
    standard_error: npt.NDArray[np.float64],
    chi_squared: npt.NDArray[np.float64],
    log10_p_value: npt.NDArray[np.float64],
    extra_code: npt.NDArray[np.int32] | None = None,
) -> None: ...
def summarize_variant_major_dosage_chunk_stats(
    genotype_matrix_by_variant: npt.NDArray[np.float32],
) -> ChunkStats: ...
def finalize_output_run_chunks(
    run_directory: str,
    chunks_directory: str,
    association_mode: g.types.AssociationMode | str,
    output_format: g.types.OutputFormat | str,
) -> str: ...
def resolve_output_run_paths(
    output_root: str,
    association_mode: g.types.AssociationMode | str,
    output_format: g.types.OutputFormat | str,
) -> NativeOutputRunPaths: ...
def prepare_output_run(
    output_root: str,
    association_mode: g.types.AssociationMode | str,
    output_format: g.types.OutputFormat | str,
    resume: bool,
) -> NativePreparedOutputRun: ...
def load_run_manifest_json(run_directory: str) -> str | None: ...
def write_run_manifest_json(run_directory: str, manifest_json: str) -> None: ...
def build_current_run_manifest_header_json(
    association_mode: str,
    association_backend_kind: str,
    bgen_path: str,
    sample_path: str | None,
    phenotype_path: str,
    phenotype_name: str,
    covariate_path: str | None,
    covariate_names: list[str],
    prediction_list_path: str,
    prediction_loco_files_json: str,
    sample_count: int,
    variant_count: int,
    chunk_size: int,
    variant_limit: int | None,
    binary_correction_plan_method: str,
    binary_correction_plan_p_threshold: float,
    binary_correction_plan_firth_se: bool,
    trusted_no_missing_diploid: bool,
    sample_key_mode: str,
    binary_kernel_config_json: str | None,
    bgen_decode_tile_variant_count: int,
    trusted_bgen_validation_mode: str,
    jax_device: str,
    jax_enable_x64: bool,
    jax_matmul_precision: str | None,
    gpu_genotype_format: str,
    score_dtype: str,
    firth_dtype: str,
    multi_phenotype_sample_mode: str,
    phenotype_compute_group_id: str | None,
    sample_set_fingerprint: str | None,
    covariate_design_fingerprint: str | None,
    prediction_alignment_fingerprint: str | None,
    output_format: str,
    finalize_parquet: bool,
    writer_thread_count: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: str,
    parquet_compression: str,
    output_statistic_dtype: str,
) -> str: ...
def build_prepared_run_manifest_header_json(prepared_run_plan_json: str) -> str: ...
def build_prepared_run_plan_json(prepared_run_plan_input_json: str) -> str: ...
def resolve_prediction_loco_paths(
    prediction_list_path: str,
    phenotype_names: list[str],
) -> list[dict[str, str]]: ...
def build_file_content_sha256_value(path: str) -> str: ...
def build_manifest_file_fingerprint_payload(
    path: str,
    include_content_hash: bool,
) -> dict[str, object]: ...
def build_manifest_file_fingerprint_mapping_payload(
    path: str,
    size: int,
    mtime_ns: int,
    content_hash_algorithm: str,
    content_sha256: str | None,
) -> dict[str, object]: ...
def build_manifest_json_sha256(manifest_json: str) -> str: ...
def validate_run_manifest_compatibility(manifest_json: str, current_header_json: str) -> None: ...
def read_manifest_committed_chunk_identifiers(manifest_json: str) -> list[int]: ...
def initialize_output_run(
    run_directory: str,
    chunks_directory: str,
    existing_manifest_json: str | None,
    current_header_json: str,
    resume: bool,
    resume_mode: g.types.ResumeMode | str,
) -> NativeInitializedOutputRun: ...
def build_logging_runtime_policy_payload(
    log_filter: str,
    log_file: str | None,
    log_stderr: bool,
    log_queue_size: int,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: str | None,
    trace_filter: str,
    trace_event_cap: int | None,
    telemetry_mode: str,
    telemetry_stream_file: str | None,
) -> dict[str, object]: ...
def describe_logging_runtime_policy_value(
    log_filter: str,
    log_file: str | None,
    log_stderr: bool,
    log_queue_size: int,
    log_lossy: bool,
    include_source_location: bool,
    include_span_events: bool,
    trace_file: str | None,
    trace_filter: str,
    trace_event_cap: int | None,
) -> str: ...
def build_shutdown_signal_payload(signal_number: int) -> dict[str, object]: ...
def configure_bgen_decode_tile_variant_count(tile_variant_count: int) -> None: ...
def configure_rayon_global_thread_pool(thread_count: int) -> None: ...
def initialize_logging(
    log_filter: str | None = None,
    log_file: str | None = None,
    log_stderr: bool = True,
    log_queue_size: int = 65536,
    log_lossy: bool = True,
    include_source_location: bool = False,
    include_span_events: bool = False,
    trace_file: str | None = None,
    trace_filter: str | None = None,
    trace_event_cap: int | None = None,
) -> bool: ...
def shutdown_logging() -> None: ...
def build_telemetry_event_payload(
    run_id: str,
    event: str,
    level: str,
    timestamp: str,
    process_identifier: int,
    thread_name: str,
    fields: dict[str, object],
) -> dict[str, object]: ...
def format_telemetry_timestamp_value(timestamp_seconds: float) -> str: ...
def resolve_telemetry_output_run_root_value(
    output_path: str,
    output_run_directory: str | None,
) -> str: ...
def resolve_telemetry_paths_payload(
    output_path: str,
    output_run_directory: str | None,
    telemetry_mode: str,
    log_dir: str | None,
    log_file: str | None,
    trace_file: str | None,
    profile_summary_json: str | None,
    stage_timings_json: str | None,
) -> dict[str, object]: ...
def resolve_telemetry_stream_file_value(
    telemetry_mode: str,
    log_dir: str | None,
    log_file: str | None,
    trace_file: str | None,
) -> str | None: ...
def paths_refer_to_same_file_value(first_path: str, second_path: str) -> bool: ...
def build_empty_telemetry_writer_counters_payload() -> dict[str, object]: ...
def build_run_completed_telemetry_fields(event: object) -> dict[str, object]: ...
def build_run_interrupted_telemetry_fields(event: object) -> dict[str, object]: ...
def build_run_failed_telemetry_fields(event: object) -> dict[str, object]: ...
def build_phenotype_run_artifacts_payload(
    output_run_directory: str,
    chunks_directory: str,
    effective_config: str,
    phenotype_name: str,
    association_mode: str,
    phenotype_count: int,
    output_format: str,
    final_output_path: str | None,
) -> dict[str, object]: ...
def build_multi_run_artifacts_payload(
    association_mode: str,
    phenotype_count: int,
) -> dict[str, object]: ...
def build_run_manifest_extension_payload(
    phenotype_name: str,
    effective_config: str,
    output_format: str,
    device: str,
    staging_depth: int,
    native_callback_batch_size: int,
    threads: int | None,
    writer_threads: int,
    writer_queue_depth: int,
    chunks_per_arrow_file: int,
    arrow_compression: str,
    parquet_compression: str,
    output_statistic_dtype: str,
    bgen_decode_tile_variant_count: int,
    trusted_no_missing_diploid: bool,
    trusted_bgen_validation_mode: str,
) -> dict[str, object]: ...
def build_trusted_bgen_validation_fingerprint_value(
    bgen_path: str,
    sample_count: int,
    variant_count: int,
    trusted_no_missing_diploid: bool,
) -> str: ...
def build_trusted_bgen_validation_cache_path_value(
    cache_directory: str,
    fingerprint: str,
) -> str: ...
def build_trusted_bgen_validation_cache_payload(
    fingerprint: str,
    bgen_path: str,
    sample_count: int,
    variant_count: int,
) -> dict[str, object]: ...
def render_run_completed_lines(event: object) -> tuple[str, ...]: ...
def render_run_interrupted_lines(event: object) -> tuple[str, ...]: ...
def render_run_failed_lines(event: object) -> tuple[str, ...]: ...
def plan_association_backend_payload(
    association_mode: str,
    jax_device: str,
    gpu_genotype_format: str,
) -> dict[str, object]: ...
def resolve_association_mode_value(trait_type: str) -> str: ...
def normalize_binary_correction_payload(
    firth: bool,
    approx: bool,
    spa: bool,
    p_threshold: float,
    firth_se: bool,
) -> dict[str, object]: ...
def build_phenotype_compute_groups_payload(
    phenotype_names: typing.Sequence[str],
    multi_phenotype_sample_mode: str,
) -> tuple[dict[str, object], ...]: ...
def build_phenotype_compute_group_id_value(
    group_mode: str,
    phenotype_indices: typing.Sequence[int],
    phenotype_names: typing.Sequence[str],
    sample_mode: str,
    sample_set_fingerprint: str | None,
    covariate_design_fingerprint: str | None,
    prediction_alignment_fingerprint: str | None,
) -> str: ...
def build_phenotype_output_directory_name(phenotype_index: int, phenotype_name: str) -> str: ...
def resolve_jax_runtime_setup_payload(
    requested_device: str,
    cache_directory: str,
    matmul_precision: str | None,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: int,
    persistent_cache_min_compile_time_seconds: int,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> dict[str, object]: ...
def scan_committed_chunk_identifiers(chunks_directory: str) -> list[int]: ...
def repair_strict_manifest_chunk_commits(chunks_directory: str, manifest_json: str) -> str: ...
def validate_strict_manifest_chunks(chunks_directory: str, manifest_json: str) -> list[int]: ...
def plan_genotype_chunks(
    variant_count: int,
    chunk_size: int,
    chromosome_boundary_indices: list[int],
    variant_limit: int | None = None,
    committed_chunk_identifiers: list[int] | None = None,
) -> list[ChunkSpec]: ...
def align_sample_data(
    sample_indices: npt.NDArray[np.int64],
    family_identifiers: list[str],
    individual_identifiers: list[str],
    phenotype_path: str,
    phenotype_name: str,
    covariate_path: str | None = None,
    covariate_names: list[str] | None = None,
    is_binary_trait: bool = False,
    sample_key_mode: g.types.SampleKeyMode | str = "iid",
) -> NativeAlignedSampleData: ...
def align_multi_sample_data(
    sample_indices: npt.NDArray[np.int64],
    family_identifiers: list[str],
    individual_identifiers: list[str],
    phenotype_path: str,
    phenotype_names: list[str],
    covariate_path: str | None = None,
    covariate_names: list[str] | None = None,
    is_binary_trait: bool = False,
    sample_key_mode: g.types.SampleKeyMode | str = "iid",
) -> NativeMultiAlignedSampleData: ...
def align_grouped_sample_data(
    sample_indices: npt.NDArray[np.int64],
    family_identifiers: list[str],
    individual_identifiers: list[str],
    phenotype_path: str,
    phenotype_names: list[str],
    covariate_path: str | None = None,
    covariate_names: list[str] | None = None,
    is_binary_trait: bool = False,
    sample_key_mode: g.types.SampleKeyMode | str = "iid",
) -> NativeGroupedAlignedSampleData: ...
def config_from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig: ...
def config_from_toml(path: str | Path) -> RegenieConfig: ...
def load_packaged_config() -> RegenieConfig: ...
def config_option_schema() -> list[dict[str, typing.Any]]: ...
def dumps_config_toml(config: RegenieConfig) -> str: ...
def write_config_toml(config: RegenieConfig, path: str | Path) -> None: ...
def validate_regenie_config(config: RegenieConfig) -> None: ...
def validate_regenie_config_for_run(config: RegenieConfig) -> None: ...
def compile_run_request_json(config: RegenieConfig) -> str: ...
def dispatch_cli(args: list[str]) -> CliOutcome: ...
def resolve_preflight_variant_count(variant_count: int, variant_limit: int | None = None) -> int: ...
def intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: typing.Sequence[typing.Sequence[int]],
) -> list[int]: ...
def resolve_bgen_delivery_method_value(
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> str: ...
def resolve_callback_worker_backpressure_poll_timeout_seconds() -> float: ...
def resolve_callback_worker_stop_poll_timeout_seconds(remaining_timeout_seconds: float) -> float: ...
def build_callback_chunk_identity(
    chromosome: str,
    variant_start_index: int,
    variant_stop_index: int,
) -> NativeCallbackChunkIdentity: ...
def should_attempt_callback_worker_stop(
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> bool: ...
def resolve_delivery_callback_batch_size(
    callback_batch_size: int | None,
    variant_major_packed8_probability_pairs: bool,
) -> int: ...
def resolve_grouped_union_callback_batch_size(native_callback_batch_size: int) -> int: ...
def resolve_native_callback_queue_limits(
    staging_depth: int,
    native_callback_batch_size: int,
    result_in_flight_limit: int | None,
    dosage_buffer_limit: int | None,
) -> NativeCallbackQueueLimits: ...
def resolve_native_callback_worker_shutdown_timeouts() -> NativeCallbackWorkerShutdownTimeouts: ...
def plan_dosage_callback_worker_join(
    timeout_seconds: float | None,
    has_started: bool,
) -> NativeCallbackWorkerJoinPlan: ...
def plan_result_callback_worker_join(
    timeout_seconds: float | None,
    has_started: bool,
) -> NativeCallbackWorkerJoinPlan: ...
def plan_dosage_callback_worker_stop(
    timeout_seconds: float | None,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> NativeCallbackWorkerStopPlan: ...
def plan_result_callback_worker_stop(
    timeout_seconds: float | None,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> NativeCallbackWorkerStopPlan: ...
def plan_callback_worker_finish() -> NativeCallbackWorkerFinishPlan: ...
def plan_callback_worker_abort() -> NativeCallbackWorkerAbortPlan: ...
def plan_callback_worker_stop_poll(
    remaining_timeout_seconds: float,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> NativeCallbackWorkerStopPollPlan: ...
def format_dosage_callback_worker_error_message(error_message: str) -> str: ...
def format_result_callback_worker_error_message(error_message: str) -> str: ...
def plan_callback_queue_operation_observation(
    queue_name: str,
    operation_name: str,
    elapsed_seconds: float,
    blocked: bool,
) -> NativeCallbackQueueOperationObservationPlan: ...
def plan_callback_queue_stage_observation(
    queue_name: str,
    operation_name: str,
    elapsed_seconds: float,
    blocked: bool,
) -> NativeCallbackQueueStageObservationPlan: ...
def plan_multi_trait_chunk_write(
    writer_session_count: int,
    chunk_identifier: int,
    committed_chunk_identifier_sets: typing.Sequence[typing.Sequence[int]],
) -> NativeMultiTraitChunkWritePlan: ...
def resolve_writer_finish_thread_count(writer_session_count: int, requested_thread_count: int) -> int: ...
def plan_writer_finish_execution(
    writer_session_count: int,
    requested_thread_count: int,
) -> NativeWriterFinishExecutionPlan: ...
def plan_single_trait_output_write(
    is_native_writer_session: bool,
    output_statistic_dtype: str,
) -> NativeSingleTraitOutputWritePlan: ...
def plan_multi_trait_output_write(
    active_trait_count: int,
    all_writer_sessions_native: bool,
    output_statistic_dtype: str,
) -> NativeMultiTraitOutputWritePlan: ...
def plan_dosage_buffer_reuse(
    buffered_shape: typing.Sequence[int],
    expected_shape: typing.Sequence[int],
) -> NativeDosageBufferReusePlan | None: ...
def plan_variant_major_dosage_batch_handoff(
    metadata_count: int,
    genotype_matrix_by_variant_count: int,
    chunk_stats_count: int,
) -> NativeVariantMajorDosageBatchHandoffPlan: ...
def build_preflight_report_payload(
    sample_count: int,
    covariate_count: int,
    chromosome_count: int,
    trusted_no_missing_diploid: bool,
) -> dict[str, object]: ...
def emit_diagnostic_event(level: str, event: str, message: str, fields_json: str | None = None) -> None: ...
def align_sample_data_from_sample_file(
    sample_path: str,
    expected_sample_count: int,
    phenotype_path: str,
    phenotype_name: str,
    covariate_path: str | None = None,
    covariate_names: list[str] | None = None,
    is_binary_trait: bool = False,
    sample_key_mode: g.types.SampleKeyMode | str = "iid",
) -> NativeAlignedSampleData: ...
def align_multi_sample_data_from_sample_file(
    sample_path: str,
    expected_sample_count: int,
    phenotype_path: str,
    phenotype_names: list[str],
    covariate_path: str | None = None,
    covariate_names: list[str] | None = None,
    is_binary_trait: bool = False,
    sample_key_mode: g.types.SampleKeyMode | str = "iid",
) -> NativeMultiAlignedSampleData: ...
def resolve_single_phenotype_compute_group(
    aligned_sample_data: NativeAlignedSampleData,
    phenotype_name: str,
    prediction_list_path: str | None,
    sample_key_mode: g.types.SampleKeyMode | str,
) -> NativeResolvedPhenotypeComputeGroup: ...
def resolve_per_phenotype_compute_group(
    aligned_sample_data: NativeMultiAlignedSampleData,
    phenotype_indices: list[int],
    phenotype_names: list[str],
    prediction_list_path: str | None,
    sample_key_mode: g.types.SampleKeyMode | str,
) -> NativeResolvedPhenotypeComputeGroup: ...
def resolve_complete_case_compute_group(
    aligned_sample_data: NativeMultiAlignedSampleData,
    phenotype_indices: list[int],
    phenotype_names: list[str],
    prediction_list_path: str | None,
    sample_key_mode: g.types.SampleKeyMode | str,
) -> NativeResolvedPhenotypeComputeGroup: ...
