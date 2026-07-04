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
    def validate_trusted_no_missing_diploid_with_default_cache(
        self,
        bgen_path: str,
        validation_mode: str,
    ) -> None: ...
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

class NativeTelemetryRunSession:
    def __init__(
        self,
        telemetry_mode: str,
        stream_file: str | None,
        progress_interval_seconds: float,
        progress_interval_chunks: int,
        queue_size: int = 65536,
        lossy: bool = True,
        trace_event_cap: int = 0,
        run_id: str | None = None,
    ) -> None: ...
    @property
    def run_id(self) -> str: ...
    @property
    def enabled(self) -> bool: ...
    @property
    def profile_enabled(self) -> bool: ...
    @property
    def event_cap(self) -> int | None: ...
    @property
    def has_native_telemetry_session(self) -> bool: ...
    def should_emit_progress(self, processed_chunk_count: int) -> bool: ...
    def emit_current_event(
        self,
        event: str,
        level: str,
        fields: dict[str, object],
    ) -> None: ...
    def emit_run_completed_event(self, event: object) -> None: ...
    def emit_run_interrupted_event(self, event: object) -> None: ...
    def emit_run_failed_event(self, event: object) -> None: ...
    def emit_run_started_event(
        self,
        association_mode: str,
        trait_type: str,
        phenotype_count: int,
        output_run_root: str,
    ) -> None: ...
    def emit_execution_plan_prepared_event(
        self,
        association_mode: str,
        trait_type: str,
        phenotype_count: int,
        chunk_size: int,
        variant_limit: int | None,
        device: str,
    ) -> None: ...
    def emit_effective_config_written_event(
        self,
        association_mode: str,
        phenotype: str,
        effective_config: str,
        output_run_directory: str,
    ) -> None: ...
    def emit_phenotype_writer_finished_event(
        self,
        association_mode: str,
        phenotype: str,
        final_output_path: str | None,
    ) -> None: ...
    def emit_multi_phenotype_writer_finished_event(
        self,
        association_mode: str,
        phenotype_count: int,
        final_output_paths: tuple[str | None, ...],
    ) -> None: ...
    def emit_single_trait_preflight_completed_event(
        self,
        association_mode: str,
        phenotype: str,
        sample_count: int,
        covariate_count: int,
        chromosome_count: int,
    ) -> None: ...
    def emit_multi_phenotype_preflight_completed_event(
        self,
        association_mode: str,
        phenotype_count: int,
        sample_count: int,
    ) -> None: ...
    def emit_sample_alignment_completed_event(
        self,
        association_mode: str,
        phenotype: str | None,
        phenotype_count: int | None,
        sample_count: int | None,
        covariate_count: int | None,
        phenotype_group_count: int | None,
    ) -> None: ...
    def emit_prediction_source_loaded_event(
        self,
        association_mode: str,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None: ...
    def emit_multi_phenotype_sample_summary_event(
        self,
        association_mode: str,
        multi_phenotype_sample_mode: str,
        sample_counts: tuple[int, ...],
        sample_set_fingerprints: tuple[str | None, ...],
        phenotype_group_count: int,
    ) -> None: ...
    def emit_gpu_genotype_format_resolved_event(
        self,
        requested_gpu_genotype_format: str,
        resolved_gpu_genotype_format: str,
        resolution_reason: str,
        fallback_error: str | None,
    ) -> None: ...
    def emit_association_backend_selected_event(
        self,
        association_mode: str,
        association_backend_kind: str,
        device: str,
        genotype_format: str,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None: ...
    def emit_bgen_engine_opened_event(
        self,
        association_mode: str,
        association_backend_kind: str,
        sample_count: int,
        variant_count: int,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None: ...
    def emit_callback_progress_event(
        self,
        progress_event: NativeCallbackProgressTelemetryEvent,
    ) -> None: ...
    def emit_binary_correction_summary_event(
        self,
        fields: dict[str, int],
    ) -> None: ...
    def emit_jax_runtime_diagnostic_event(
        self,
        event: object,
        telemetry_level: str,
    ) -> None: ...
    def emit_progress(
        self,
        processed_chunk_count: int,
        fields: dict[str, object],
    ) -> None: ...
    def build_current_event_payload(
        self,
        event: str,
        level: str,
        fields: dict[str, object],
    ) -> dict[str, object]: ...
    def emit_payload(self, payload: dict[str, object]) -> None: ...
    def counters(self) -> dict[str, object]: ...
    def close_metadata(self) -> dict[str, object] | None: ...
    def finish_close_metadata(self) -> dict[str, object] | None: ...
    def finish_with_current_close_event_metadata(self) -> dict[str, object] | None: ...

class NativeTelemetryClosePolicy:
    def __init__(self) -> None: ...
    def close_telemetry_session_with_event(self, telemetry_session: object | None) -> None: ...

class NativeTelemetrySessionPolicy:
    def __init__(self, telemetry_mode: str, trace_event_cap: int) -> None: ...
    @property
    def enabled(self) -> bool: ...
    @property
    def profile_enabled(self) -> bool: ...
    @property
    def event_cap(self) -> int | None: ...
    def resolve_output_run_root_value(
        self,
        output_path: str,
        output_run_directory: str | None,
    ) -> str: ...
    def resolve_paths_payload(
        self,
        output_path: str,
        output_run_directory: str | None,
        log_dir: str | None,
        log_file: str | None,
        trace_file: str | None,
        profile_summary_json: str | None,
        stage_timings_json: str | None,
    ) -> dict[str, object]: ...

class NativeBinaryCorrectionDiagnosticsRecordPlan:
    @property
    def should_record(self) -> bool: ...

class NativeBinaryCorrectionSummaryEmitPlan:
    @property
    def should_flush_pending_diagnostics(self) -> bool: ...
    @property
    def should_emit_summary(self) -> bool: ...

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
    def chunk_count_with_pending(self, pending_diagnostics_count: int) -> int: ...
    def plan_diagnostics_record(
        self,
        has_telemetry_session: bool,
        has_diagnostics: bool,
    ) -> NativeBinaryCorrectionDiagnosticsRecordPlan: ...
    def plan_summary_emit(
        self,
        has_telemetry_session: bool,
        pending_diagnostics_count: int,
    ) -> NativeBinaryCorrectionSummaryEmitPlan: ...
    def summary_payload(self) -> dict[str, int]: ...

class NativeBinaryCorrectionSummaryTelemetryPolicy:
    def __init__(self) -> None: ...
    def emit_binary_correction_summary_telemetry(
        self,
        telemetry_session: object | None,
        summary_payload: dict[str, int] | None,
        missing_session_message: str,
    ) -> None: ...

class NativeCallbackObjectQueueGetResult:
    @property
    def has_item(self) -> bool: ...
    @property
    def item(self) -> object | None: ...

class NativeCallbackQueueOperationOutcome:
    @property
    def item(self) -> object | None: ...
    @property
    def should_retry(self) -> bool: ...
    @property
    def should_stop(self) -> bool: ...
    @property
    def should_flush_binary_correction_diagnostics(self) -> bool: ...
    @property
    def dispatch_action(self) -> str: ...
    @property
    def should_process_sample_major_dosage(self) -> bool: ...
    @property
    def should_process_variant_major_dosage(self) -> bool: ...
    @property
    def should_process_variant_major_dosage_batch(self) -> bool: ...
    @property
    def should_process_variant_major_packed8_probability_pair(self) -> bool: ...
    @property
    def should_process_result_write_item(self) -> bool: ...
    @property
    def should_process_multi_result_write_item(self) -> bool: ...
    @property
    def has_dispatch_error(self) -> bool: ...
    @property
    def dispatch_error_message(self) -> str | None: ...
    @property
    def worker_error_raise_plan(self) -> NativeCallbackWorkerErrorRaisePlan | None: ...
    @property
    def stage_backpressure_observation(self) -> NativeCallbackQueueStageBackpressureObservation | None: ...

class NativeCallbackResourceOperationOutcome:
    @property
    def dosage_buffer(self) -> object | None: ...
    @property
    def should_retry(self) -> bool: ...
    @property
    def should_allocate(self) -> bool: ...
    @property
    def released_host_buffer(self) -> bool: ...
    @property
    def released_result_in_flight_slot(self) -> bool: ...
    @property
    def free_buffer_count(self) -> int | None: ...
    @property
    def backpressure_observation(self) -> NativeCallbackQueueBackpressureObservation | None: ...
    @property
    def dosage_buffer_pool_backpressure_observation(self) -> NativeCallbackQueueBackpressureObservation | None: ...
    @property
    def result_in_flight_backpressure_observation(self) -> NativeCallbackQueueBackpressureObservation | None: ...
    @property
    def worker_error_raise_plan(self) -> NativeCallbackWorkerErrorRaisePlan | None: ...
    @property
    def stage_backpressure_observation(self) -> NativeCallbackQueueStageBackpressureObservation | None: ...

class NativeDosageWorkItemStageDurationAttribution:
    @property
    def metadata_items(self) -> tuple[object, ...]: ...
    @property
    def stage_duration_plan(self) -> NativeDosageWorkItemStageDurationPlan: ...

class NativeCallbackObjectQueue:
    def __init__(self, capacity: int) -> None: ...
    @property
    def capacity(self) -> int: ...
    @property
    def occupied_count(self) -> int: ...
    @property
    def has_available_slot(self) -> bool: ...
    @property
    def has_queued_item(self) -> bool: ...
    def put(self, item: object, timeout_seconds: float) -> bool: ...
    def get(self, timeout_seconds: float) -> NativeCallbackObjectQueueGetResult: ...
    def wait_for_available_slot(self, timeout_seconds: float) -> bool: ...
    def wait_for_queued_item(self, timeout_seconds: float) -> bool: ...

class NativeCallbackWaitSignal:
    def __init__(self) -> None: ...
    @property
    def generation(self) -> int: ...
    def notify_waiters(self) -> int: ...
    def wait_for_change(self, observed_generation: int, timeout_seconds: float) -> bool: ...

class NativeCallbackWorkerThread:
    def __init__(
        self,
        *,
        target: typing.Callable[[], object],
        name: str,
        daemon: bool = True,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    def start(self) -> None: ...
    def join(self, timeout: float | None = None) -> None: ...
    def is_alive(self) -> bool: ...

class NativeCallbackWorkerFinishLifecycleResult:
    @property
    def has_shutdown_timeout(self) -> bool: ...
    @property
    def shutdown_worker_name(self) -> str | None: ...
    @property
    def shutdown_timeout_seconds(self) -> float | None: ...
    @property
    def raise_worker_error(self) -> bool: ...
    @property
    def complete_progress(self) -> bool: ...
    @property
    def progress_completion_event(self) -> NativeCallbackProgressTelemetryEvent | None: ...
    @property
    def emit_binary_correction_summary(self) -> bool: ...
    @property
    def flush_binary_correction_pending_diagnostics(self) -> bool: ...
    @property
    def binary_correction_summary_payload(self) -> dict[str, int] | None: ...

class NativeCallbackRuntimeResources:
    def __init__(
        self,
        *,
        worker_name: str,
        dosage_worker_target: typing.Callable[[], object],
        result_worker_target: typing.Callable[[], object],
        staging_depth: int,
        native_callback_batch_size: int,
        expected_result_work_item_kind: str,
        has_telemetry_session: bool,
        flush_binary_correction_diagnostics_on_result_stop: bool,
        has_stage_timing_recorder: bool = False,
        result_in_flight_limit: int | None = None,
        dosage_buffer_limit: int | None = None,
    ) -> None: ...
    @property
    def dosage_worker_name(self) -> str: ...
    @property
    def result_worker_name(self) -> str: ...
    @property
    def dosage_worker_is_alive(self) -> bool: ...
    @property
    def result_worker_is_alive(self) -> bool: ...
    @property
    def has_started(self) -> bool: ...
    @property
    def native_callback_batch_size(self) -> int: ...
    @property
    def dosage_queue_depth(self) -> int: ...
    @property
    def result_queue_depth(self) -> int: ...
    @property
    def result_in_flight_limit(self) -> int: ...
    @property
    def dosage_buffer_limit(self) -> int: ...
    @property
    def dosage_queue_occupied_count(self) -> int: ...
    @property
    def result_queue_occupied_count(self) -> int: ...
    @property
    def result_in_flight_occupied_count(self) -> int: ...
    @property
    def dosage_buffer_allocated_count(self) -> int: ...
    @property
    def free_dosage_buffer_count(self) -> int: ...
    @property
    def dosage_buffer_identifiers(self) -> list[int]: ...
    @property
    def processed_chunk_count(self) -> int: ...
    @property
    def current_progress_chromosome(self) -> str | None: ...
    def record_processed_chunk(self, chunk_identity: NativeCallbackChunkIdentity) -> NativeCallbackProgressUpdate: ...
    def record_processed_chunk_for_metadata(self, metadata: object) -> NativeCallbackProgressUpdate: ...
    def record_progress_for_metadata(self, metadata: object) -> NativeCallbackProgressUpdate | None: ...
    def record_processed_chunk_without_progress(self) -> None: ...
    def finish_progress(self) -> NativeCallbackProgressCompletion | None: ...
    def binary_correction_chunk_count_with_pending(self, pending_diagnostics_count: int) -> int: ...
    def binary_correction_chunk_count_with_pending_diagnostics(self, pending_diagnostics: object) -> int: ...
    def add_binary_null_model_failure_count(self, failure_count: int) -> None: ...
    def plan_binary_correction_diagnostics_record(
        self,
        has_diagnostics: bool,
    ) -> NativeBinaryCorrectionDiagnosticsRecordPlan: ...
    def plan_binary_correction_diagnostics_record_for_object(
        self,
        binary_chunk_diagnostics: object | None,
    ) -> NativeBinaryCorrectionDiagnosticsRecordPlan: ...
    def plan_binary_correction_summary_emit(
        self,
        pending_diagnostics_count: int,
    ) -> NativeBinaryCorrectionSummaryEmitPlan: ...
    def plan_binary_correction_summary_emit_for_pending_diagnostics(
        self,
        pending_diagnostics: object,
    ) -> NativeBinaryCorrectionSummaryEmitPlan: ...
    def add_binary_correction_diagnostics_totals(
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
    def binary_correction_summary_payload(self) -> dict[str, int]: ...
    def start_workers(self) -> NativeCallbackWorkerStartAttemptPlan: ...
    def stop_dosage_worker(self, timeout_seconds: float | None) -> float | None: ...
    def join_dosage_worker(self, timeout_seconds: float | None) -> float | None: ...
    def stop_result_worker(self, timeout_seconds: float | None) -> float | None: ...
    def join_result_worker(self, timeout_seconds: float | None) -> float | None: ...
    def finish_worker_lifecycle(
        self,
        pending_diagnostics_count: int,
    ) -> NativeCallbackWorkerFinishLifecycleResult: ...
    def finish_worker_lifecycle_for_pending_diagnostics(
        self,
        pending_diagnostics: object,
    ) -> NativeCallbackWorkerFinishLifecycleResult: ...
    def abort_worker_lifecycle(self) -> NativeCallbackWorkerAbortPlan: ...
    def plan_worker_error_raise(self) -> NativeCallbackWorkerErrorRaisePlan: ...
    def update_dosage_worker_error(self, error_message: str | None) -> NativeCallbackWorkerErrorUpdatePlan: ...
    def update_result_worker_error(self, error_message: str | None) -> NativeCallbackWorkerErrorUpdatePlan: ...
    def put_dosage_work_item_outcome(self, work_item: object) -> NativeCallbackQueueOperationOutcome: ...
    def put_dosage_work_item_until_accepted_outcome(
        self,
        work_item: object,
    ) -> NativeCallbackQueueOperationOutcome: ...
    def put_result_write_item_outcome(self, work_item: object) -> NativeCallbackQueueOperationOutcome: ...
    def put_result_write_item_until_accepted_outcome(
        self,
        work_item: object,
    ) -> NativeCallbackQueueOperationOutcome: ...
    def get_next_dosage_work_item_outcome(self) -> NativeCallbackQueueOperationOutcome: ...
    def get_next_result_write_item_outcome(self) -> NativeCallbackQueueOperationOutcome: ...
    def acquire_result_in_flight_slot_outcome(self) -> NativeCallbackResourceOperationOutcome: ...
    def acquire_result_in_flight_slot_until_available_outcome(self) -> NativeCallbackResourceOperationOutcome: ...
    def release_result_in_flight_slot_outcome(self) -> NativeCallbackResourceOperationOutcome: ...
    def acquire_dosage_buffer_outcome(self) -> NativeCallbackResourceOperationOutcome: ...
    def acquire_reusable_dosage_buffer_or_allocate_outcome(
        self,
        expected_shape: typing.Sequence[int],
        expected_dtype: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def register_dosage_buffer_outcome(
        self,
        dosage_buffer: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def release_dosage_buffer_outcome(
        self,
        dosage_buffer: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def release_numpy_dosage_buffer_outcome(
        self,
        dosage_buffer: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def discard_dosage_buffer_outcome(
        self,
        dosage_buffer: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def select_reusable_dosage_buffer_or_discard_outcome(
        self,
        dosage_buffer: object,
        expected_shape: typing.Sequence[int],
        expected_dtype: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def release_result_work_item_resources_outcome(
        self,
        work_item: object,
        phase: str,
        host_dosage_buffer_released: bool,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def release_result_work_item_pre_write_resources_outcome(
        self,
        work_item: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def release_result_work_item_final_resources_outcome(
        self,
        work_item: object,
        host_dosage_buffer_released: bool,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def release_result_work_item_in_flight_slot_outcome(
        self,
        work_item: object,
    ) -> NativeCallbackResourceOperationOutcome: ...
    def get_releasable_dosage_buffer_owner(
        self,
        dosage_buffer: object,
    ) -> object | None: ...
    def plan_dosage_work_item_stage_duration(
        self,
        dosage_work_item_kind: str,
        chunk_count: int,
        elapsed_seconds: float,
    ) -> NativeDosageWorkItemStageDurationPlan: ...
    def plan_dosage_work_item_stage_duration_for_object(
        self,
        work_item: object,
        elapsed_seconds: float,
    ) -> NativeDosageWorkItemStageDurationPlan: ...
    def plan_dosage_work_item_stage_duration_attribution_for_object(
        self,
        work_item: object,
        elapsed_seconds: float,
    ) -> NativeDosageWorkItemStageDurationAttribution: ...
    def plan_current_queue_backpressure_observation(
        self,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueBackpressureObservation: ...
    def plan_current_queue_stage_backpressure_observation(
        self,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueStageBackpressureObservation: ...
    def plan_dosage_buffer_pool_backpressure_observation(
        self,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueBackpressureObservation: ...
    def plan_dosage_buffer_pool_stage_backpressure_observation(
        self,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueStageBackpressureObservation: ...
    def plan_variant_major_dosage_batch_handoff(
        self,
        metadata_count: int,
        genotype_matrix_by_variant_count: int,
        chunk_stats_count: int,
    ) -> NativeVariantMajorDosageBatchHandoffPlan: ...
    def plan_variant_major_dosage_batch_handoff_for_sequences(
        self,
        metadata_batch: object,
        genotype_matrix_by_variant_batch: object,
        chunk_stats_batch: object,
    ) -> NativeVariantMajorDosageBatchHandoffPlan: ...
    def plan_dosage_work_handoff(self, chunk_count: int) -> NativeDosageWorkHandoffPlan: ...
    def plan_dosage_work_handoff_for_object(
        self,
        work_item: object,
    ) -> NativeDosageWorkHandoffPlan: ...

class NativeCallbackQueueBackpressureObservation:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def queue_capacity(self) -> int: ...
    @property
    def elapsed_seconds(self) -> float: ...
    @property
    def blocked_seconds(self) -> float: ...

class NativeCallbackQueueStageBackpressureObservation:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def stage_name(self) -> str: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def queue_capacity(self) -> int: ...
    @property
    def elapsed_seconds(self) -> float: ...
    @property
    def blocked_seconds(self) -> float: ...

class NativeDosageWorkItemStageDurationPlan:
    @property
    def chunk_count(self) -> int: ...
    @property
    def duration_per_chunk(self) -> float: ...

class NativeCallbackWorkerErrorRaisePlan:
    @property
    def should_raise(self) -> bool: ...
    @property
    def raise_dosage_worker_error(self) -> bool: ...
    @property
    def raise_result_worker_error(self) -> bool: ...
    @property
    def error_message(self) -> str | None: ...

class NativeCallbackWorkerErrorUpdatePlan:
    @property
    def had_error(self) -> bool: ...
    @property
    def has_error(self) -> bool: ...
    @property
    def error_message(self) -> str | None: ...

class NativeCallbackWorkerStartAttemptPlan:
    @property
    def start_actions(self) -> list[str]: ...
    @property
    def should_start(self) -> bool: ...
    @property
    def start_result_worker(self) -> bool: ...
    @property
    def start_dosage_worker(self) -> bool: ...
    @property
    def has_marked_started(self) -> bool: ...
    @property
    def has_start_error(self) -> bool: ...
    @property
    def error_message(self) -> str | None: ...

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
    @property
    def telemetry_plan(self) -> NativeCallbackProgressTelemetryPlan: ...

class NativeCallbackProgressTelemetryEvent:
    @property
    def event_name(self) -> str: ...
    @property
    def level(self) -> str: ...
    @property
    def chromosome(self) -> str: ...
    @property
    def processed_chunk_count(self) -> int: ...

class NativeCallbackProgressTelemetryRecord:
    @property
    def processed_chunk_count(self) -> int: ...
    @property
    def chromosome(self) -> str: ...
    @property
    def chunk_identifier(self) -> int: ...
    @property
    def variant_start_index(self) -> int: ...
    @property
    def variant_stop_index(self) -> int: ...
    @property
    def variant_count(self) -> int: ...

class NativeCallbackProgressTelemetryPlan:
    @property
    def events(self) -> list[NativeCallbackProgressTelemetryEvent]: ...
    @property
    def progress(self) -> NativeCallbackProgressTelemetryRecord: ...

class NativeCallbackProgressCompletion:
    @property
    def chromosome(self) -> str: ...
    @property
    def processed_chunk_count(self) -> int: ...
    @property
    def telemetry_event(self) -> NativeCallbackProgressTelemetryEvent: ...

class NativeCallbackProgressState:
    def __init__(self) -> None: ...
    @property
    def processed_chunk_count(self) -> int: ...
    @property
    def current_progress_chromosome(self) -> str | None: ...
    def record_processed_chunk(self, chunk_identity: NativeCallbackChunkIdentity) -> NativeCallbackProgressUpdate: ...
    def record_processed_chunk_without_progress(self) -> None: ...
    def finish_progress(self) -> NativeCallbackProgressCompletion | None: ...

class NativeCallbackProgressPolicy:
    def __init__(self) -> None: ...
    def build_callback_chunk_identity(
        self,
        chromosome: str,
        variant_start_index: int,
        variant_stop_index: int,
    ) -> NativeCallbackChunkIdentity: ...
    def emit_callback_progress_update_telemetry(
        self,
        telemetry_session: object | None,
        progress_update: NativeCallbackProgressUpdate | None,
    ) -> None: ...
    def emit_callback_progress_event_telemetry(
        self,
        telemetry_session: object | None,
        progress_event: NativeCallbackProgressTelemetryEvent | None,
        missing_session_message: str,
    ) -> None: ...
    def emit_callback_progress_completion_telemetry(
        self,
        telemetry_session: object | None,
        progress_completion: NativeCallbackProgressCompletion | None,
    ) -> None: ...

class NativeVariantMajorDosageBatchHandoffPlan:
    @property
    def chunk_count(self) -> int: ...

class NativeDosageWorkHandoffPlan:
    @property
    def chunk_count(self) -> int: ...

class NativeGpuGenotypeFormatResolutionPlan:
    @property
    def requested_gpu_genotype_format(self) -> str: ...
    @property
    def resolved_gpu_genotype_format(self) -> str | None: ...
    @property
    def resolution_reason(self) -> str | None: ...
    @property
    def fallback_error(self) -> str | None: ...
    @property
    def requires_trusted_validation(self) -> bool: ...
    @property
    def is_resolved(self) -> bool: ...
    @property
    def should_log_auto_resolution(self) -> bool: ...

class NativeNullLogisticNonconvergencePlan:
    @property
    def action(self) -> str: ...
    @property
    def failed_trait_indices(self) -> list[int]: ...
    @property
    def message(self) -> str | None: ...
    @property
    def warning_message(self) -> str | None: ...
    @property
    def nonconverged_count(self) -> int: ...
    @property
    def scalar_convergence(self) -> bool: ...
    @property
    def total_fit_count(self) -> int: ...

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

class NativeBgenDeliveryCleanupPlan:
    @property
    def cleanup_actions(self) -> list[str]: ...
    @property
    def drain_callback(self) -> bool: ...
    @property
    def finish_writer_sessions(self) -> bool: ...
    @property
    def finish_interrupted_writer_sessions(self) -> bool: ...
    @property
    def abort_callback(self) -> bool: ...
    @property
    def abort_writer_sessions(self) -> bool: ...
    @property
    def write_stage_timing_snapshot(self) -> bool: ...

class NativeBgenDeliveryInvocationPlan:
    @property
    def delivery_method(self) -> str: ...
    @property
    def callback_batch_size(self) -> int: ...

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

def intersect_committed_chunk_identifier_sets(
    committed_chunk_identifier_sets: typing.Sequence[typing.Sequence[int]],
) -> list[int]: ...
def resolve_delivery_callback_batch_size(
    callback_batch_size: int | None,
    variant_major_packed8_probability_pairs: bool,
) -> int: ...
def resolve_grouped_union_callback_batch_size(native_callback_batch_size: int) -> int: ...
def resolve_manifest_gpu_genotype_format(
    resume: bool,
    manifest_gpu_genotype_format: str | None,
    association_backend_genotype_format: str | None,
) -> str | None: ...
def resolve_effective_trusted_no_missing_diploid(
    requested_trusted_no_missing_diploid: bool,
    variant_major_packed8_probability_pairs: bool,
) -> bool: ...
def plan_gpu_genotype_format_auto_to_dosage(
    requested_gpu_genotype_format: str,
    resolution_reason: str,
) -> NativeGpuGenotypeFormatResolutionPlan: ...
def plan_single_trait_binary_gpu_genotype_format_resolution(
    requested_gpu_genotype_format: str,
    manifest_gpu_genotype_format: str | None,
    association_backend_genotype_format: str | None,
    resume: bool,
    jax_device: str,
) -> NativeGpuGenotypeFormatResolutionPlan: ...
def plan_auto_gpu_genotype_format_after_trusted_validation(
    fallback_error: str | None,
) -> NativeGpuGenotypeFormatResolutionPlan: ...
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
def plan_bgen_delivery_cleanup(
    cleanup_outcome: str,
    callback_finished: bool,
) -> NativeBgenDeliveryCleanupPlan: ...
def plan_bgen_delivery_invocation(
    callback_batch_size: int | None,
    variant_major_packed8_probability_pairs: bool,
    has_native_multi_aligned_sample_data: bool,
    has_native_aligned_sample_data: bool,
) -> NativeBgenDeliveryInvocationPlan: ...
def plan_single_trait_output_write(
    is_native_writer_session: bool,
    output_statistic_dtype: str,
) -> NativeSingleTraitOutputWritePlan: ...
def plan_multi_trait_output_write(
    active_trait_count: int,
    all_writer_sessions_native: bool,
    output_statistic_dtype: str,
) -> NativeMultiTraitOutputWritePlan: ...

class NativeCallbackWorkerLifecycleState:
    def __init__(self) -> None: ...
    @property
    def has_started(self) -> bool: ...
    def mark_started(self) -> bool: ...

class NativeCallbackWorkerStartPlan:
    @property
    def start_actions(self) -> list[str]: ...
    @property
    def should_start(self) -> bool: ...
    @property
    def start_result_worker(self) -> bool: ...
    @property
    def start_dosage_worker(self) -> bool: ...

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
    def finish_actions(self) -> list[str]: ...
    @property
    def stop_dosage_worker(self) -> bool: ...
    @property
    def join_dosage_worker(self) -> bool: ...
    @property
    def stop_result_worker(self) -> bool: ...
    @property
    def join_result_worker(self) -> bool: ...
    @property
    def raise_worker_error(self) -> bool: ...
    @property
    def complete_progress(self) -> bool: ...
    @property
    def emit_binary_correction_summary(self) -> bool: ...
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
    def abort_actions(self) -> list[str]: ...
    @property
    def stop_dosage_worker(self) -> bool: ...
    @property
    def stop_result_worker(self) -> bool: ...
    @property
    def dosage_stop_timeout_seconds(self) -> float: ...
    @property
    def result_stop_timeout_seconds(self) -> float: ...

class NativeCallbackWorkerStopPollPlan:
    @property
    def should_stop(self) -> bool: ...
    @property
    def poll_timeout_seconds(self) -> float: ...

class NativeStageTimingRecorder:
    exact_stage_timings: bool
    def __init__(self, exact_stage_timings: bool) -> None: ...
    @staticmethod
    def from_config(stage_timing_path_configured: bool, force: bool) -> NativeStageTimingRecorder | None: ...
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
    def add_scalar_null_logistic_diagnostics_from_arrays(
        self,
        chromosome: str,
        convergence_values: object,
        iteration_count_values: object,
        firth_iteration_count_values: object,
        firth_convergence_reason_code_values: object,
        correction_method: str,
    ) -> None: ...
    def add_multi_null_logistic_diagnostics_from_arrays(
        self,
        chromosome: str,
        convergence_values: object,
        iteration_count_values: object,
        phenotype_names: typing.Sequence[str],
        correction_method: str,
    ) -> None: ...
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
    def add_transfer_metadata_for_shape(
        self,
        transfer_name: str,
        array_role: str,
        dtype_name: str,
        shape_dimensions: typing.Sequence[int],
        item_size: int,
    ) -> None: ...
    def snapshot_payload(self) -> dict[str, object]: ...
    def write_final_timing_outputs(
        self,
        stage_timing_path: str | None,
        profile_summary_path: str | None,
        run_id: str | None,
    ) -> dict[str, bool]: ...

class NativeFinalTimingOutputContext:
    stage_timing_path: str | None
    profile_summary_path: str | None
    run_id: str | None
    force_stage_timing_recorder: bool

class NativeFinalTimingOutputPolicy:
    def __init__(self) -> None: ...
    def resolve_final_timing_output_context(
        self,
        diagnostics_stage_timing_path: str | None,
        telemetry_session: object | None,
    ) -> NativeFinalTimingOutputContext: ...
    def record_final_timing_outputs_write_started_diagnostic_event(
        self,
        stage_timing_path: str | None,
        profile_summary_path: str | None,
        run_id: str | None,
    ) -> None: ...

class NativeCliTelemetryCloseFailurePlan:
    should_report_failure: bool
    exit_code: int

class NativeCliRunLifecycleState:
    def __init__(self) -> None: ...
    @property
    def runner_started(self) -> bool: ...
    def mark_runner_started(self) -> None: ...
    def emit_run_failed_telemetry_event(
        self,
        telemetry_session: object | None,
        failed_event: object,
    ) -> None: ...
    def plan_telemetry_close_failure(
        self,
        current_exit_code: int,
        runtime_failure_exit_code: int,
    ) -> NativeCliTelemetryCloseFailurePlan: ...

class NativeRuntimeCompatibilityToken:
    pass

class NativeLoggingRuntimePolicy:
    @property
    def log_filter(self) -> str: ...
    @property
    def log_file(self) -> str | None: ...
    @property
    def log_stderr(self) -> bool: ...
    @property
    def log_queue_size(self) -> int: ...
    @property
    def log_lossy(self) -> bool: ...
    @property
    def include_source_location(self) -> bool: ...
    @property
    def include_span_events(self) -> bool: ...
    @property
    def trace_file(self) -> str | None: ...
    @property
    def trace_filter(self) -> str: ...
    @property
    def trace_event_cap(self) -> int | None: ...

class NativeJaxRuntimePolicy:
    @property
    def device(self) -> str: ...
    @property
    def cache_directory(self) -> str | None: ...
    @property
    def matmul_precision(self) -> str | None: ...
    @property
    def persistent_cache(self) -> bool: ...
    @property
    def persistent_cache_min_entry_size_bytes(self) -> int: ...
    @property
    def persistent_cache_min_compile_time_seconds(self) -> int: ...
    @property
    def xla_autotune_cache(self) -> bool: ...
    @property
    def transfer_guard(self) -> bool: ...

class NativeRuntimeStateSnapshot:
    @property
    def logging_policy(self) -> NativeLoggingRuntimePolicy | None: ...
    @property
    def rayon_thread_count(self) -> int | None: ...
    @property
    def jax_policy(self) -> NativeJaxRuntimePolicy | None: ...

class NativeRuntimePolicy:
    rayon_thread_count: int | None
    def logging_runtime_policy(self) -> NativeLoggingRuntimePolicy: ...
    def jax_runtime_policy(self) -> NativeJaxRuntimePolicy: ...

class NativeRunRuntime:
    rayon_thread_count: int | None
    def logging_runtime_policy(self) -> NativeLoggingRuntimePolicy: ...
    def jax_runtime_policy(self) -> NativeJaxRuntimePolicy: ...
    def runtime_compatibility_token(self) -> NativeRuntimeCompatibilityToken: ...

class NativeRayonThreadPoolConfigurationPlan:
    should_configure: bool
    thread_count: int | None

class NativeJaxRuntimeSetupLifecyclePlan:
    should_configure: bool

class NativeJaxRuntimeDiagnosticRecordPlan:
    logging_level_name: str
    should_emit_telemetry: bool
    telemetry_level: str

class NativeJaxRuntimeDiagnosticPolicy:
    def __init__(self) -> None: ...
    def record_jax_runtime_diagnostic_event(
        self,
        event: object,
        telemetry_session: object | None,
    ) -> NativeJaxRuntimeDiagnosticRecordPlan: ...

class NativeJaxRuntimeSetupReport:
    @property
    def requested_device(self) -> str: ...
    @property
    def platform_name(self) -> str: ...
    @property
    def cache_directory(self) -> str: ...
    @property
    def matmul_precision(self) -> str: ...
    @property
    def persistent_cache_enabled(self) -> bool: ...
    @property
    def persistent_cache_min_entry_size_bytes(self) -> int: ...
    @property
    def persistent_cache_min_compile_time_seconds(self) -> int: ...
    @property
    def xla_auxiliary_cache_mode(self) -> str: ...
    @property
    def xla_auxiliary_cache_reason(self) -> str: ...
    @property
    def transfer_guard_enabled(self) -> bool: ...
    @property
    def gpu_validation_status(self) -> str: ...
    @property
    def gpu_validation_message(self) -> str | None: ...

class NativeNvidiaDriverProbePaths:
    @property
    def control_device_path(self) -> str: ...
    @property
    def uvm_device_path(self) -> str: ...
    @property
    def driver_directory_path(self) -> str: ...

class NativeJaxRuntimeDiagnosticField:
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> object: ...

class NativeJaxRuntimeDiagnosticEvent:
    @property
    def event_name(self) -> str: ...
    @property
    def level(self) -> str: ...
    @property
    def message(self) -> str: ...
    @property
    def fields(self) -> list[NativeJaxRuntimeDiagnosticField]: ...

class NativeJaxRuntimeSetupSession:
    @property
    def should_configure(self) -> bool: ...
    @property
    def should_validate_gpu(self) -> bool: ...
    def setup_report(self) -> NativeJaxRuntimeSetupReport: ...
    def apply_config_updates(self) -> int: ...
    def complete_validation_report(
        self,
        gpu_validation_status: str,
        gpu_validation_message: str | None,
    ) -> NativeJaxRuntimeSetupReport: ...
    def diagnostic_events(self) -> list[NativeJaxRuntimeDiagnosticEvent]: ...
    def create_cache_directory_if_configured(self) -> bool: ...
    def validate_gpu_if_configured(
        self,
        control_device_path: str,
        uvm_device_path: str,
        driver_directory_path: str,
    ) -> NativeJaxRuntimeSetupReport: ...
    def validate_gpu_if_configured_with_default_probe_paths(self) -> NativeJaxRuntimeSetupReport: ...
    def nvidia_driver_files_are_visible(
        self,
        control_device_path: str,
        uvm_device_path: str,
        driver_directory_path: str,
    ) -> bool: ...
    def nvidia_driver_files_are_visible_with_default_probe_paths(self) -> bool: ...
    def default_nvidia_driver_probe_paths(self) -> NativeNvidiaDriverProbePaths: ...

class NativeRuntimeState:
    rayon_thread_count: int | None
    def __init__(self) -> None: ...
    @staticmethod
    def global_process_runtime_state() -> NativeRuntimeState: ...
    def logging_runtime_policy(self) -> NativeLoggingRuntimePolicy | None: ...
    def jax_runtime_policy(self) -> NativeJaxRuntimePolicy | None: ...
    def default_local_cache_directory_value(self, directory_name: str) -> str: ...
    def describe_logging_runtime_policy_value(
        self,
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
    def build_logging_runtime_policy(
        self,
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
    ) -> NativeLoggingRuntimePolicy: ...
    def build_jax_runtime_policy(
        self,
        device: str,
        cache_directory: str | None,
        matmul_precision: str | None,
        persistent_cache: bool,
        persistent_cache_min_entry_size_bytes: int,
        persistent_cache_min_compile_time_seconds: int,
        xla_autotune_cache: bool,
        transfer_guard: bool,
    ) -> NativeJaxRuntimePolicy: ...
    def build_runtime_policy_handle(
        self,
        logging_policy: NativeLoggingRuntimePolicy,
        rayon_thread_count: int | None,
        jax_policy: NativeJaxRuntimePolicy,
    ) -> NativeRuntimePolicy: ...
    def build_process_runtime_state_handle(
        self,
        logging_policy: NativeLoggingRuntimePolicy | None,
        rayon_thread_count: int | None,
        jax_policy: NativeJaxRuntimePolicy | None,
    ) -> NativeRuntimeState: ...
    def runtime_state_snapshot(self) -> NativeRuntimeStateSnapshot: ...
    def require_compatible_runtime_policy(
        self,
        logging_policy: NativeLoggingRuntimePolicy,
        rayon_thread_count: int | None,
        jax_policy: NativeJaxRuntimePolicy,
    ) -> NativeRuntimeCompatibilityToken: ...
    def build_run_runtime(
        self,
        runtime_policy: NativeRuntimePolicy,
    ) -> NativeRunRuntime: ...
    def require_compatible_runtime_policy_handle(
        self,
        runtime_policy: NativeRuntimePolicy,
    ) -> NativeRuntimeCompatibilityToken: ...
    def require_compatible_logging_runtime_policy(self, logging_policy: NativeLoggingRuntimePolicy) -> None: ...
    def record_logging_runtime_policy(self, logging_policy: NativeLoggingRuntimePolicy) -> None: ...
    def initialize_logging_runtime_policy(self, logging_policy: NativeLoggingRuntimePolicy) -> bool: ...
    def shutdown_logging_runtime(self) -> None: ...
    def require_compatible_rayon_thread_count(self, thread_count: int | None) -> None: ...
    def record_rayon_thread_count(self, thread_count: int) -> None: ...
    def plan_rayon_thread_pool_configuration(self, thread_count: int) -> NativeRayonThreadPoolConfigurationPlan: ...
    def configure_rayon_thread_pool(self, thread_count: int) -> NativeRayonThreadPoolConfigurationPlan: ...
    def configure_runtime_knobs(
        self,
        bgen_decode_tile_variant_count: int,
        rayon_thread_count: int | None,
    ) -> NativeRayonThreadPoolConfigurationPlan | None: ...
    def effective_rayon_thread_count(self, requested_thread_count: int | None) -> int | None: ...
    def require_compatible_jax_runtime_policy(self, jax_policy: NativeJaxRuntimePolicy) -> None: ...
    def record_jax_runtime_policy(self, jax_policy: NativeJaxRuntimePolicy) -> None: ...
    def complete_jax_runtime_setup(self, jax_policy: NativeJaxRuntimePolicy) -> None: ...
    def complete_jax_runtime_setup_session(
        self,
        jax_policy: NativeJaxRuntimePolicy,
        setup_session: NativeJaxRuntimeSetupSession,
    ) -> None: ...
    def plan_jax_runtime_setup_lifecycle(
        self,
        jax_policy: NativeJaxRuntimePolicy,
    ) -> NativeJaxRuntimeSetupLifecyclePlan: ...
    def build_jax_runtime_setup_session(
        self,
        jax_policy: NativeJaxRuntimePolicy,
        resolved_cache_directory: str,
    ) -> NativeJaxRuntimeSetupSession: ...
    def build_jax_runtime_setup_session_resolving_cache_directory(
        self,
        jax_policy: NativeJaxRuntimePolicy,
    ) -> NativeJaxRuntimeSetupSession: ...

class NativeShutdownController:
    def __init__(self, handled_signal_numbers: typing.Sequence[int] | None = None) -> None: ...
    @property
    def handlers_installed(self) -> bool: ...
    def reset(self) -> None: ...
    def requested_signal_payload(self) -> dict[str, object] | None: ...
    def request_shutdown_payload(self, signal_number: int) -> dict[str, object]: ...
    def request_shutdown_signal_or_raise_second_signal_payload(self, signal_number: int) -> dict[str, object]: ...
    def handler_install_plan_payload(self) -> dict[str, object]: ...
    def mark_handlers_installed(self) -> None: ...
    def install_python_signal_handlers(self, handler: object) -> None: ...
    def handler_restore_plan_payload(self) -> dict[str, object]: ...
    def mark_handlers_restored(self) -> None: ...
    def restore_python_signal_handlers(self) -> bool: ...
    def restore_python_signal_handlers_and_reset(self) -> bool: ...

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
    def existing_manifest_payload(self) -> dict[str, object] | None: ...

class NativeInitializedOutputRun:
    @property
    def committed_chunk_identifiers(self) -> list[int]: ...

class NativeManifestFileFingerprint:
    @property
    def path(self) -> str: ...
    @property
    def size(self) -> int: ...
    @property
    def mtime_ns(self) -> int: ...
    @property
    def content_hash_algorithm(self) -> str: ...
    @property
    def content_sha256(self) -> str | None: ...

class NativePredictionLocoFileFingerprint:
    @property
    def phenotype(self) -> str: ...
    @property
    def path(self) -> str: ...
    @property
    def size(self) -> int: ...
    @property
    def mtime_ns(self) -> int: ...
    @property
    def content_hash_algorithm(self) -> str: ...
    @property
    def content_sha256(self) -> str | None: ...

class NativeOutputLifecyclePolicy:
    def __init__(self) -> None: ...
    def finalize_output_run_chunks(
        self,
        run_directory: str,
        chunks_directory: str,
        association_mode: g.types.AssociationMode | str,
        output_format: g.types.OutputFormat | str,
    ) -> str: ...
    def resolve_output_run_paths(
        self,
        output_root: str,
        association_mode: g.types.AssociationMode | str,
        output_format: g.types.OutputFormat | str,
    ) -> NativeOutputRunPaths: ...
    def prepare_output_run(
        self,
        output_root: str,
        association_mode: g.types.AssociationMode | str,
        output_format: g.types.OutputFormat | str,
        resume: bool,
        runtime_compatibility_token: NativeRuntimeCompatibilityToken,
    ) -> NativePreparedOutputRun: ...
    def load_run_manifest_payload(self, run_directory: str) -> dict[str, object] | None: ...
    def write_run_manifest(self, run_directory: str, manifest: object) -> None: ...
    def build_prepared_run_plan_json_from_current_header(self, current_header: object) -> str: ...
    def build_manifest_json_sha256_from_value(self, value: object) -> str: ...
    def validate_run_manifest_compatibility_from_values(self, manifest: object, current_header: object) -> None: ...
    def read_manifest_committed_chunk_identifiers_from_value(self, manifest: object) -> list[int]: ...
    def initialize_output_run_from_values(
        self,
        run_directory: str,
        chunks_directory: str,
        existing_manifest: object | None,
        current_header: object,
        resume: bool,
        resume_mode: g.types.ResumeMode | str,
        runtime_compatibility_token: NativeRuntimeCompatibilityToken,
    ) -> NativeInitializedOutputRun: ...
    def scan_committed_chunk_identifiers(self, chunks_directory: str) -> list[int]: ...
    def repair_strict_manifest_chunk_commits_from_value(
        self,
        chunks_directory: str,
        manifest: object,
    ) -> tuple[dict[str, object], ...]: ...
    def validate_strict_manifest_chunks_from_value(self, chunks_directory: str, manifest: object) -> list[int]: ...

class NativeManifestFileFingerprintCache:
    def __init__(self) -> None: ...
    def build_file_fingerprint(
        self,
        path: str,
        include_content_hash: bool,
    ) -> NativeManifestFileFingerprint: ...
    def build_current_run_manifest_header_payload_from_input(
        self,
        current_header_input: object,
    ) -> dict[str, object]: ...
    def build_prediction_loco_file_fingerprints(
        self,
        prediction_list_path: str,
        phenotype_names: list[str],
    ) -> list[NativePredictionLocoFileFingerprint]: ...

class NativeOutputChunkWritePolicy:
    def __init__(self) -> None: ...
    def write_regenie2_multi_native_chunk(
        self,
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
        self,
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

class NativeRunEventPayloadPolicy:
    def __init__(self) -> None: ...
    def attach_run_metadata_payload(
        self,
        artifacts: object,
        run_id: str | None,
        association_mode: str,
        phenotype_count: int,
    ) -> dict[str, object]: ...
    def build_run_completed_event_payload(self, artifacts: object) -> dict[str, object]: ...
    def build_run_interrupted_event_payload(self, shutdown_request: object) -> dict[str, object]: ...
    def build_run_failed_event_payload(self, error: BaseException) -> dict[str, object]: ...
    def render_run_completed_lines(self, event: object) -> tuple[str, ...]: ...
    def render_run_interrupted_lines(self, event: object) -> tuple[str, ...]: ...
    def render_run_failed_lines(self, event: object) -> tuple[str, ...]: ...

class NativeRunEventTelemetryPolicy:
    def __init__(self) -> None: ...
    def record_runner_run_started_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        trait_type: str,
        phenotype_count: int,
        output_run_root: str,
    ) -> None: ...
    def record_runner_run_interrupted_telemetry_event(
        self,
        telemetry_session: object | None,
        event: object,
    ) -> None: ...
    def record_runner_run_failed_telemetry_event(
        self,
        telemetry_session: object | None,
        event: object,
    ) -> None: ...
    def record_runner_run_completed_telemetry_event(
        self,
        telemetry_session: object | None,
        event: object,
    ) -> None: ...
    def record_execution_plan_prepared_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        trait_type: str,
        phenotype_count: int,
        chunk_size: int,
        variant_limit: int | None,
        device: str,
    ) -> None: ...
    def record_effective_config_written_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        phenotype: str,
        effective_config: str,
        output_run_directory: str,
    ) -> None: ...
    def record_writer_finished_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        phenotype: str,
        final_output_path: str | None,
    ) -> None: ...
    def record_multi_writer_finished_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        phenotype_count: int,
        final_output_paths: typing.Sequence[str | None],
    ) -> None: ...
    def record_single_trait_preflight_completed_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        phenotype: str,
        sample_count: int,
        covariate_count: int,
        chromosome_count: int,
    ) -> None: ...
    def record_multi_phenotype_preflight_completed_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        phenotype_count: int,
        sample_count: int,
    ) -> None: ...
    def record_sample_alignment_completed_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        phenotype: str | None,
        phenotype_count: int | None,
        sample_count: int | None,
        covariate_count: int | None,
        phenotype_group_count: int | None,
    ) -> None: ...
    def record_prediction_source_loaded_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None: ...
    def record_multi_phenotype_sample_summary_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        sample_mode: str,
        sample_counts: typing.Sequence[int],
        sample_set_fingerprints: typing.Sequence[str | None],
        phenotype_group_count: int,
    ) -> None: ...
    def record_gpu_genotype_format_resolved_telemetry_event(
        self,
        telemetry_session: object | None,
        requested_gpu_genotype_format: str,
        resolved_gpu_genotype_format: str,
        resolution_reason: str,
        fallback_error: str | None,
    ) -> None: ...
    def record_association_backend_selected_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        association_backend_kind: str,
        device: str,
        genotype_format: str,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None: ...
    def record_bgen_engine_opened_telemetry_event(
        self,
        telemetry_session: object | None,
        association_mode: str,
        association_backend_kind: str,
        sample_count: int,
        variant_count: int,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None: ...

class NativeCliDiagnosticPolicy:
    def __init__(self) -> None: ...
    def record_native_cli_stdout_diagnostic_event(
        self,
        output_text: str,
        max_payload_chars: int,
    ) -> None: ...
    def record_native_cli_stderr_diagnostic_event(
        self,
        output_text: str,
        max_payload_chars: int,
    ) -> None: ...
    def record_native_cli_interrupted_line_diagnostic_event(
        self,
        line: str,
    ) -> None: ...
    def record_native_cli_failed_line_diagnostic_event(
        self,
        line: str,
    ) -> None: ...
    def record_native_cli_completed_line_diagnostic_event(
        self,
        line: str,
    ) -> None: ...
    def record_native_runtime_knobs_configured_diagnostic_event(
        self,
        bgen_decode_tile_variant_count: int,
        threads: int | None,
    ) -> None: ...

class NativeRunnerDiagnosticPolicy:
    def __init__(self) -> None: ...
    def record_runner_run_started_diagnostic_event(
        self,
        association_mode: str,
        trait_type: str,
        phenotype_count: int,
    ) -> None: ...
    def record_runner_run_interrupted_diagnostic_event(self, event: object) -> None: ...
    def record_runner_run_failed_diagnostic_event(self, event: object) -> None: ...
    def record_runner_run_completed_diagnostic_event(self, event: object) -> None: ...
    def record_runner_jax_runtime_configuration_started_diagnostic_event(self) -> None: ...
    def record_runner_execution_plan_build_started_diagnostic_event(self) -> None: ...
    def record_runner_execution_plan_prepared_diagnostic_event(
        self,
        association_mode: str,
        phenotype_count: int,
        chunk_size: int,
        variant_limit: int | None,
        device: str,
    ) -> None: ...
    def record_runner_execution_plan_dispatch_started_diagnostic_event(
        self,
        phenotype_count: int,
        association_mode: str,
    ) -> None: ...
    def record_runner_execution_plan_finalization_started_diagnostic_event(
        self,
        phenotype_count: int,
        association_mode: str,
    ) -> None: ...
    def record_runner_multi_phenotype_dispatch_started_diagnostic_event(
        self,
        phenotype_count: int,
        association_mode: str,
    ) -> None: ...
    def record_runner_single_phenotype_dispatch_started_diagnostic_event(
        self,
        association_mode: str,
        phenotype: str,
    ) -> None: ...
    def record_runner_binary_engine_dispatch_started_diagnostic_event(
        self,
        phenotype: str,
    ) -> None: ...
    def record_runner_linear_engine_dispatch_started_diagnostic_event(
        self,
        phenotype: str,
    ) -> None: ...
    def record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event(
        self,
        phenotype_count: int,
    ) -> None: ...
    def record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event(
        self,
        phenotype_count: int,
    ) -> None: ...
    def record_runner_metadata_artifacts_finalized_diagnostic_event(
        self,
        association_mode: str,
        phenotype_count: int,
    ) -> None: ...

class NativeOutputPreflightDiagnosticPolicy:
    def __init__(self) -> None: ...
    def record_preflight_warning_diagnostic_event(
        self,
        message: str,
        chromosome_count: int,
        covariate_count: int,
        preflight_scope: str,
        sample_count: int,
        trusted_no_missing_diploid: bool,
        warning_index: int,
    ) -> None: ...
    def record_io_output_resume_committed_chunks_diagnostic_event(
        self,
        chunks_directory: str,
        committed_chunk_count: int,
        run_directory: str,
    ) -> None: ...

class NativePipelineDiagnosticPolicy:
    def __init__(self) -> None: ...
    def record_pipeline_bgen_engine_open_started_diagnostic_event(
        self,
        phenotype_count: int | None,
        phenotype_name: str | None,
        pipeline_label: str,
        trusted_no_missing_diploid: bool,
        variant_limit: int | None,
    ) -> None: ...
    def record_pipeline_bgen_engine_opened_diagnostic_event(
        self,
        phenotype_count: int | None,
        phenotype_name: str | None,
        pipeline_label: str,
        sample_count: int,
        variant_count: int,
    ) -> None: ...
    def record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
        self,
        phenotype_count: int | None,
        phenotype_name: str | None,
        pipeline_label: str,
    ) -> None: ...
    def record_pipeline_output_resume_committed_chunks_diagnostic_event(
        self,
        committed_chunk_count: int,
        output_index: int,
    ) -> None: ...
    def record_pipeline_output_writer_sessions_create_started_diagnostic_event(
        self,
        association_mode: str,
        output_count: int,
    ) -> None: ...
    def record_pipeline_gpu_genotype_format_resolved_diagnostic_event(
        self,
        requested_gpu_genotype_format: str,
        resolved_gpu_genotype_format: str,
        resolution_reason: str,
        fallback_error: str | None,
    ) -> None: ...
    def record_callback_null_logistic_nonconvergence_warning_diagnostic_event(
        self,
        message: str,
        chromosome: str,
        nonconverged_count: int,
        phenotype_count: int,
        policy: str,
        scalar_convergence: bool,
        total_fit_count: int,
    ) -> None: ...
    def record_pipeline_multi_phenotype_sample_summary_diagnostic_event(
        self,
        phenotype_count: int,
        phenotype_group_count: int,
        sample_counts_differ: bool,
        sample_mode: str,
    ) -> None: ...
    def record_pipeline_multi_trait_started_diagnostic_event(
        self,
        association_mode: str,
        phenotype_count: int,
        sample_mode: str,
    ) -> None: ...
    def record_pipeline_multi_trait_input_load_started_diagnostic_event(
        self,
        phenotype_count: int,
    ) -> None: ...
    def record_pipeline_multi_trait_input_aligned_diagnostic_event(
        self,
        covariate_count: int,
        phenotype_count: int,
        sample_count: int,
    ) -> None: ...
    def record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event(
        self,
        phenotype_count: int,
    ) -> None: ...
    def record_pipeline_grouped_per_phenotype_started_diagnostic_event(
        self,
        association_mode: str,
        phenotype_count: int,
        sample_mode: str,
    ) -> None: ...
    def record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
        self,
        phenotype_count: int,
        phenotype_group_count: int,
    ) -> None: ...
    def record_pipeline_grouped_union_delivery_selected_diagnostic_event(
        self,
        grouped_sample_count: int,
        phenotype_group_count: int,
        union_sample_count: int,
    ) -> None: ...
    def record_pipeline_multi_group_preflight_started_diagnostic_event(
        self,
        phenotype_count: int,
        sample_count: int,
        trusted_no_missing_diploid: bool,
        variant_limit: int | None,
    ) -> None: ...
    def record_pipeline_multi_group_preflight_completed_diagnostic_event(
        self,
        phenotype_count: int,
        sample_count: int,
        trusted_no_missing_diploid: bool,
        variant_limit: int | None,
    ) -> None: ...
    def record_pipeline_single_trait_started_diagnostic_event(
        self,
        association_mode: str,
        phenotype_name: str,
        pipeline_label: str,
    ) -> None: ...
    def record_pipeline_single_trait_input_load_started_diagnostic_event(
        self,
        phenotype_name: str,
        pipeline_label: str,
    ) -> None: ...
    def record_pipeline_single_trait_input_aligned_diagnostic_event(
        self,
        covariate_count: int,
        phenotype_name: str,
        pipeline_label: str,
        sample_count: int,
    ) -> None: ...
    def record_pipeline_single_trait_prediction_source_load_started_diagnostic_event(
        self,
        phenotype_name: str,
        pipeline_label: str,
    ) -> None: ...
    def record_pipeline_single_trait_preflight_started_diagnostic_event(
        self,
        phenotype_name: str,
        pipeline_label: str,
        trusted_no_missing_diploid: bool,
        variant_limit: int | None,
    ) -> None: ...
    def record_pipeline_single_trait_preflight_completed_diagnostic_event(
        self,
        chromosome_count: int,
        covariate_count: int,
        phenotype_name: str,
        pipeline_label: str,
        sample_count: int,
    ) -> None: ...

class NativeDispatchDiagnosticPolicy:
    def __init__(self) -> None: ...
    def record_native_dispatch_bgen_engine_constructing_diagnostic_event(
        self,
        chunk_size: int,
        source_path: str,
        trusted_no_missing_diploid: bool,
        variant_limit: int | None,
    ) -> None: ...
    def record_native_dispatch_trusted_bgen_validation_started_diagnostic_event(
        self,
        source_path: str,
        trusted_bgen_validation_mode: str,
    ) -> None: ...
    def record_native_dispatch_delivery_started_diagnostic_event(
        self,
        committed_chunk_count: int,
        pipeline_label: str,
        variant_major_packed8_probability_pairs: bool,
    ) -> None: ...
    def record_native_dispatch_delivery_finished_diagnostic_event(
        self,
        pipeline_label: str,
        processed_chunk_count: int,
    ) -> None: ...
    def record_native_dispatch_delivery_interrupted_diagnostic_event(
        self,
        pipeline_label: str,
        signal_exit_code: int,
        signal_name: str,
        signal_number: int,
    ) -> None: ...
    def record_native_dispatch_delivery_failed_diagnostic_event(
        self,
        exception_message: str,
        exception_type: str,
        pipeline_label: str,
    ) -> None: ...
    def record_native_dispatch_pipeline_finished_diagnostic_event(
        self,
        final_parquet_path_count: int,
        pipeline_label: str,
    ) -> None: ...
    def record_native_dispatch_callback_drain_started_diagnostic_event(self) -> None: ...
    def record_native_dispatch_writer_session_finish_started_diagnostic_event(self) -> None: ...
    def record_native_dispatch_writer_sessions_finish_started_diagnostic_event(
        self,
        requested_thread_count: int,
        writer_session_count: int,
    ) -> None: ...
    def record_native_dispatch_writer_session_interrupted_flush_started_diagnostic_event(
        self,
        signal_exit_code: int,
        signal_name: str,
        signal_number: int,
    ) -> None: ...
    def record_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_event(
        self,
        requested_thread_count: int,
        signal_exit_code: int,
        signal_name: str,
        signal_number: int,
        writer_session_count: int,
    ) -> None: ...

class NativeRunMetadataBuilder:
    def __init__(self) -> None: ...
    def build_execution_run_artifacts_payload(
        self,
        association_mode: str,
        phenotype_count: int,
        output_format: str,
        output_run_directories: tuple[str, ...],
        chunks_directories: tuple[str, ...],
        effective_configs: tuple[str, ...],
        phenotype_names: tuple[str, ...],
        final_output_paths: tuple[str | None, ...],
    ) -> dict[str, object]: ...
    def extend_run_manifest_metadata(
        self,
        run_directory: str,
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
    ) -> None: ...

class NativeAssociationBackendPlan:
    @property
    def backend_kind(self) -> str: ...
    @property
    def association_mode(self) -> str: ...
    @property
    def jax_device(self) -> str: ...
    @property
    def genotype_format(self) -> str: ...
    @property
    def uses_variant_major_packed8_delivery(self) -> bool: ...

class NativeHostPlanningPolicy:
    def __init__(self) -> None: ...
    def plan_association_backend(
        self,
        association_mode: str,
        jax_device: str,
        gpu_genotype_format: str,
    ) -> NativeAssociationBackendPlan: ...
    def resolve_association_mode_value(self, trait_type: str) -> str: ...
    def normalize_binary_correction_payload(
        self,
        firth: bool,
        approx: bool,
        spa: bool,
        p_threshold: float,
        firth_se: bool,
    ) -> dict[str, object]: ...
    def build_phenotype_compute_groups_payload(
        self,
        phenotype_names: typing.Sequence[str],
        multi_phenotype_sample_mode: str,
    ) -> tuple[dict[str, object], ...]: ...
    def build_phenotype_compute_group_id_value(
        self,
        group_mode: str,
        phenotype_indices: typing.Sequence[int],
        phenotype_names: typing.Sequence[str],
        sample_mode: str,
        sample_set_fingerprint: str | None,
        covariate_design_fingerprint: str | None,
        prediction_alignment_fingerprint: str | None,
    ) -> str: ...
    def build_phenotype_output_directory_name(self, phenotype_index: int, phenotype_name: str) -> str: ...

def config_from_options(raw_options: typing.Mapping[str, typing.Any]) -> RegenieConfig: ...
def config_from_toml(path: str | Path) -> RegenieConfig: ...
def load_packaged_config() -> RegenieConfig: ...
def config_option_schema() -> list[dict[str, typing.Any]]: ...
def dumps_config_toml(config: RegenieConfig) -> str: ...
def write_config_toml(config: RegenieConfig, path: str | Path) -> None: ...
def validate_regenie_config(config: RegenieConfig) -> None: ...
def validate_regenie_config_for_run(config: RegenieConfig) -> None: ...
def compile_run_request_payload(config: RegenieConfig) -> dict[str, object]: ...
def dispatch_cli(args: list[str]) -> CliOutcome: ...
def run_native_cli_python_bridge(
    args: list[str],
    python_executable_path: str | Path,
    sentinel_environment_variable: str,
) -> CliOutcome: ...

class NativePreflightReport:
    @property
    def sample_count(self) -> int: ...
    @property
    def covariate_count(self) -> int: ...
    @property
    def chromosome_count(self) -> int: ...
    @property
    def warning_messages(self) -> list[str]: ...

class NativeSingleTraitPreflightShape:
    @property
    def sample_count(self) -> int: ...
    @property
    def covariate_count(self) -> int: ...

class NativeMultiTraitPreflightShape:
    @property
    def trait_count(self) -> int: ...
    @property
    def sample_count(self) -> int: ...
    @property
    def covariate_count(self) -> int: ...

class NativePreflightValidator:
    def __init__(self) -> None: ...
    def resolve_preflight_variant_count(self, variant_count: int, variant_limit: int | None = None) -> int: ...
    def build_preflight_report(
        self,
        sample_count: int,
        covariate_count: int,
        chromosome_count: int,
        trusted_no_missing_diploid: bool,
    ) -> NativePreflightReport: ...
    def validate_single_trait_preflight_shape(
        self,
        phenotype_sample_count: int,
        covariate_dimension_count: int,
        covariate_sample_count: int,
        covariate_count: int,
    ) -> NativeSingleTraitPreflightShape: ...
    def validate_multi_trait_preflight_shape(
        self,
        phenotype_dimension_count: int,
        phenotype_trait_count: int,
        phenotype_sample_count: int,
        covariate_dimension_count: int,
        covariate_sample_count: int,
        covariate_count: int,
    ) -> NativeMultiTraitPreflightShape: ...
    def validate_binary_phenotype_array(self, phenotype_values: object) -> None: ...
    def validate_finite_array_values(self, label: str, values: object) -> None: ...
    def validate_covariate_matrix_rank(self, covariate_rank: int, covariate_count: int) -> None: ...
    def validate_covariate_matrix_rank_array(self, covariate_matrix: object, covariate_count: int) -> None: ...
    def validate_single_prediction_preflight_shape(
        self,
        chromosome: str,
        prediction_shape: typing.Sequence[int],
        sample_count: int,
    ) -> None: ...
    def validate_multi_prediction_preflight_shape(
        self,
        chromosome: str,
        prediction_shape: typing.Sequence[int],
        trait_count: int,
        sample_count: int,
    ) -> None: ...

class NativeCallbackDiagnosticsPolicy:
    def __init__(self) -> None: ...
    def plan_null_logistic_nonconvergence_from_array(
        self,
        chromosome: str,
        convergence_values: object,
        phenotype_names: typing.Sequence[str] | None,
        policy: str,
    ) -> NativeNullLogisticNonconvergencePlan: ...

class NativePipelineOutputInitialization:
    @property
    def output_count(self) -> int: ...
    def committed_chunk_identifier_sets(self) -> list[list[int]]: ...
    def committed_chunk_identifiers(self, output_index: int) -> list[int]: ...

class NativePipelineOutputPreparationBatch:
    @property
    def output_count(self) -> int: ...
    @property
    def resume(self) -> bool: ...
    def validate_resume_compatibility(self) -> None: ...
    def initialize(
        self,
        runtime_compatibility_token: NativeRuntimeCompatibilityToken,
    ) -> NativePipelineOutputInitialization: ...

class NativePipelineOutputPreparationPolicy:
    def __init__(self) -> None: ...
    def build_pipeline_output_preparation_batch_from_values(
        self,
        run_directories: typing.Sequence[str],
        chunks_directories: typing.Sequence[str],
        existing_manifest_values: typing.Sequence[object | None],
        current_header_values: typing.Sequence[object],
        resume: bool,
        resume_mode: str,
    ) -> NativePipelineOutputPreparationBatch: ...

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
