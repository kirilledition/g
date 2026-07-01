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

class NativePreparedGroupInput:
    def __init__(self, group_identifier: str, phenotype_count: int) -> None: ...
    @property
    def group_identifier(self) -> str: ...
    @property
    def phenotype_count(self) -> int: ...

class NativePredictionView:
    def __init__(self, chromosome: str, row_count: int) -> None: ...
    @property
    def chromosome(self) -> str: ...
    @property
    def row_count(self) -> int: ...

class NativeGenotypeBatchView:
    def __init__(self, chromosome: str, variant_count: int, variant_offset: int) -> None: ...
    @property
    def chromosome(self) -> str: ...
    @property
    def variant_count(self) -> int: ...
    @property
    def variant_offset(self) -> int: ...

class NativeAssociationChromosomeRunInput:
    def __init__(
        self,
        chromosome: str,
        prediction_chromosome: str,
        prediction_row_count: int,
        batches: typing.Sequence[NativeGenotypeBatchView],
    ) -> None: ...
    @property
    def chromosome(self) -> str: ...
    @property
    def prediction_chromosome(self) -> str: ...
    @property
    def prediction_row_count(self) -> int: ...
    @property
    def batches(self) -> list[NativeGenotypeBatchView]: ...

class NativeAssociationBatchResult:
    def __init__(self, chromosome: str, variant_count: int, statistic_sum: float) -> None: ...
    @property
    def chromosome(self) -> str: ...
    @property
    def variant_count(self) -> int: ...
    @property
    def statistic_sum(self) -> float: ...

class NativeAssociationEngineRunReport:
    @property
    def phase_history(self) -> list[str]: ...
    @property
    def result(self) -> NativeAssociationBatchResult: ...

class NativeAssociationChromosomeRunReport:
    @property
    def phase_history(self) -> list[str]: ...
    @property
    def results(self) -> list[NativeAssociationBatchResult]: ...

class NativeAssociationGroupRunReport:
    @property
    def phase_history(self) -> list[str]: ...
    @property
    def results(self) -> list[NativeAssociationBatchResult]: ...

class NativePythonEngineRunEffects:
    def __init__(self, effects: object) -> None: ...

class NativePythonAssociationBackend:
    def __init__(self, backend: object) -> None: ...
    def prepare_group(self, group_identifier: str, phenotype_count: int) -> object: ...
    def prepare_chromosome(
        self,
        group_state: object,
        chromosome: str,
        prediction_chromosome: str,
        prediction_row_count: int,
    ) -> object: ...
    def compute_batch(
        self,
        chromosome_state: object,
        batch_chromosome: str,
        variant_count: int,
        variant_offset: int,
    ) -> NativeAssociationBatchResult: ...
    def run_single_batch(
        self,
        group_identifier: str,
        phenotype_count: int,
        chromosome: str,
        prediction_chromosome: str,
        prediction_row_count: int,
        batch_chromosome: str,
        variant_count: int,
        variant_offset: int,
    ) -> NativeAssociationEngineRunReport: ...
    def run_single_batch_with_effects(
        self,
        group_identifier: str,
        phenotype_count: int,
        chromosome: str,
        prediction_chromosome: str,
        prediction_row_count: int,
        batch_chromosome: str,
        variant_count: int,
        variant_offset: int,
        effects: NativePythonEngineRunEffects,
    ) -> NativeAssociationEngineRunReport: ...
    def run_chromosome_batches(
        self,
        group_identifier: str,
        phenotype_count: int,
        chromosome: str,
        prediction_chromosome: str,
        prediction_row_count: int,
        batches: typing.Sequence[NativeGenotypeBatchView],
    ) -> NativeAssociationChromosomeRunReport: ...
    def run_chromosome_batches_with_effects(
        self,
        group_identifier: str,
        phenotype_count: int,
        chromosome: str,
        prediction_chromosome: str,
        prediction_row_count: int,
        batches: typing.Sequence[NativeGenotypeBatchView],
        effects: NativePythonEngineRunEffects,
    ) -> NativeAssociationChromosomeRunReport: ...
    def run_group_chromosomes(
        self,
        group_identifier: str,
        phenotype_count: int,
        chromosome_inputs: typing.Sequence[NativeAssociationChromosomeRunInput],
    ) -> NativeAssociationGroupRunReport: ...
    def run_group_chromosomes_with_effects(
        self,
        group_identifier: str,
        phenotype_count: int,
        chromosome_inputs: typing.Sequence[NativeAssociationChromosomeRunInput],
        effects: NativePythonEngineRunEffects,
    ) -> NativeAssociationGroupRunReport: ...

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
    def validate_trusted_no_missing_diploid_with_cache(
        self,
        bgen_path: str,
        validation_mode: str,
        cache_directory: str,
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

class NativeTelemetryProgressThrottle:
    def __init__(self, progress_interval_seconds: float, progress_interval_chunks: int) -> None: ...
    def should_emit_progress(self, processed_chunk_count: int) -> bool: ...

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

class NativeTelemetryEventEmissionPlan:
    @property
    def should_emit(self) -> bool: ...

class NativeTelemetryProgressEmissionPlan:
    @property
    def should_emit(self) -> bool: ...
    @property
    def event_name(self) -> str: ...
    @property
    def level(self) -> str: ...

class NativeTelemetryClosePlan:
    @property
    def should_close(self) -> bool: ...
    @property
    def use_native_close_with_event(self) -> bool: ...
    @property
    def should_emit_legacy_close_event(self) -> bool: ...
    @property
    def legacy_close_event_name(self) -> str: ...
    @property
    def legacy_close_event_level(self) -> str: ...

class NativeTelemetrySessionPolicy:
    def __init__(self, telemetry_mode: str, trace_event_cap: int) -> None: ...
    @property
    def enabled(self) -> bool: ...
    @property
    def profile_enabled(self) -> bool: ...
    @property
    def event_cap(self) -> int | None: ...

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
    def close_metadata(self) -> dict[str, object] | None: ...
    def finish(self) -> dict[str, object]: ...
    def finish_close_metadata(self) -> dict[str, object]: ...
    def finish_with_close_event(
        self,
        run_id: str,
        timestamp: str,
        process_identifier: int,
        thread_name: str,
    ) -> dict[str, object]: ...
    def finish_with_close_event_metadata(
        self,
        run_id: str,
        timestamp: str,
        process_identifier: int,
        thread_name: str,
    ) -> dict[str, object]: ...
    def finish_with_current_close_event(self, run_id: str) -> dict[str, object]: ...
    def finish_with_current_close_event_metadata(self, run_id: str) -> dict[str, object]: ...

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

def emit_binary_correction_summary_telemetry(
    telemetry_session: object | None,
    summary_payload: dict[str, int] | None,
    missing_session_message: str,
) -> None: ...

class NativeCallbackQueueLimits:
    @property
    def dosage_queue_depth(self) -> int: ...
    @property
    def result_queue_depth(self) -> int: ...
    @property
    def result_in_flight_limit(self) -> int: ...
    @property
    def dosage_buffer_limit(self) -> int: ...

class NativeCallbackObjectQueueGetResult:
    @property
    def has_item(self) -> bool: ...
    @property
    def item(self) -> object | None: ...

class NativeCallbackQueueGetObservedResult:
    @property
    def has_item(self) -> bool: ...
    @property
    def item(self) -> object | None: ...
    @property
    def observation_plan(self) -> NativeCallbackQueueGetObservationPlan: ...

class NativeDosageWorkItemDrainResult:
    @property
    def has_dosage_work_item(self) -> bool: ...
    @property
    def item(self) -> object | None: ...
    @property
    def drain_completion_plan(self) -> NativeDosageWorkDrainCompletionPlan: ...

class NativeDosageWorkItemGetResult:
    @property
    def has_dosage_work_item(self) -> bool: ...
    @property
    def item(self) -> object | None: ...
    @property
    def observation_plan(self) -> NativeCallbackQueueGetObservationPlan | None: ...
    @property
    def stage_backpressure_observation(self) -> NativeCallbackQueueStageBackpressureObservation | None: ...
    @property
    def drain_completion_plan(self) -> NativeDosageWorkDrainCompletionPlan: ...
    @property
    def dispatch_plan(self) -> NativeDosageWorkItemDispatchPlan | None: ...

class NativeResultWriteItemDrainResult:
    @property
    def has_result_work_item(self) -> bool: ...
    @property
    def item(self) -> object | None: ...
    @property
    def drain_completion_plan(self) -> NativeResultWriteDrainCompletionPlan: ...

class NativeResultWriteItemGetResult:
    @property
    def has_result_work_item(self) -> bool: ...
    @property
    def item(self) -> object | None: ...
    @property
    def observation_plan(self) -> NativeCallbackQueueGetObservationPlan | None: ...
    @property
    def stage_backpressure_observation(self) -> NativeCallbackQueueStageBackpressureObservation | None: ...
    @property
    def drain_completion_plan(self) -> NativeResultWriteDrainCompletionPlan: ...
    @property
    def dispatch_plan(self) -> NativeResultWriteItemDispatchPlan | None: ...

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

class NativeDosageBufferAcquireResult:
    @property
    def dosage_buffer(self) -> object | None: ...
    @property
    def should_allocate(self) -> bool: ...
    @property
    def free_buffer_count(self) -> int: ...
    @property
    def waited(self) -> bool: ...
    @property
    def observation_plan(self) -> NativeDosageBufferPoolObservationPlan | None: ...
    @property
    def stage_backpressure_observation(self) -> NativeCallbackQueueStageBackpressureObservation | None: ...

class NativeDosageBufferPoolOperationResult:
    @property
    def has_free_buffer_count(self) -> bool: ...
    @property
    def free_buffer_count(self) -> int | None: ...
    @property
    def observation_plan(self) -> NativeDosageBufferPoolObservationPlan | None: ...
    @property
    def backpressure_observation(self) -> NativeCallbackQueueBackpressureObservation | None: ...

class NativeDosageBufferReuseSelectionResult:
    @property
    def dosage_buffer(self) -> object | None: ...
    @property
    def operation_result(self) -> NativeDosageBufferPoolOperationResult: ...
    @property
    def reuse_operation_result(self) -> NativeDosageBufferPoolOperationResult | None: ...
    @property
    def discard_operation_result(self) -> NativeDosageBufferPoolOperationResult | None: ...

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

class NativeResultWorkItemResourceReleaseResult:
    @property
    def released_host_buffer(self) -> bool: ...
    @property
    def free_buffer_count(self) -> int | None: ...
    @property
    def dosage_buffer_pool_observation_plan(self) -> NativeDosageBufferPoolObservationPlan | None: ...
    @property
    def dosage_buffer_pool_backpressure_observation(self) -> NativeCallbackQueueBackpressureObservation | None: ...
    @property
    def released_result_in_flight_slot(self) -> bool: ...
    @property
    def result_in_flight_observation_plan(self) -> NativeResultInFlightReleaseObservationPlan | None: ...
    @property
    def result_in_flight_backpressure_observation(self) -> NativeCallbackQueueBackpressureObservation | None: ...
    @property
    def result_in_flight_resource_name(self) -> str | None: ...
    @property
    def result_in_flight_operation_name(self) -> str | None: ...
    @property
    def result_in_flight_blocked(self) -> bool | None: ...

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
    def callback_scheduler_state(self) -> NativeCallbackSchedulerState: ...
    @property
    def progress_state(self) -> NativeCallbackProgressState: ...
    @property
    def result_in_flight_slot_signal(self) -> NativeCallbackWaitSignal: ...
    @property
    def dosage_buffer_pool_signal(self) -> NativeCallbackWaitSignal: ...
    @property
    def dosage_queue(self) -> NativeCallbackObjectQueue: ...
    @property
    def result_queue(self) -> NativeCallbackObjectQueue: ...
    @property
    def free_dosage_buffers(self) -> NativeCallbackObjectQueue: ...
    @property
    def binary_correction_summary(self) -> NativeBinaryCorrectionSummary: ...
    @property
    def worker_thread(self) -> NativeCallbackWorkerThread: ...
    @property
    def result_worker_thread(self) -> NativeCallbackWorkerThread: ...
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
    def acquire_result_in_flight_slot_with_backpressure_timeout(
        self,
    ) -> NativeResultInFlightAcquireObservationPlan: ...
    def acquire_result_in_flight_slot_with_backpressure_timeout_without_observation(
        self,
    ) -> NativeResultInFlightAcquireResult: ...
    def acquire_result_in_flight_slot_with_optional_observation(
        self,
    ) -> NativeResultInFlightAcquireResult: ...
    def release_result_in_flight_slot(self) -> NativeResultInFlightReleaseObservationPlan: ...
    def release_result_in_flight_slot_with_optional_observation(
        self,
    ) -> NativeResultInFlightReleaseObservationPlan | None: ...
    def release_result_in_flight_slot_with_optional_backpressure_observation(
        self,
    ) -> NativeResultInFlightSlotReleaseResult: ...
    def release_result_work_item_pre_write_resources(
        self,
        host_dosage_buffer: object | None,
    ) -> NativeResultWorkItemResourceReleaseResult: ...
    def release_result_work_item_pre_write_resources_for_object(
        self,
        work_item: object,
    ) -> NativeResultWorkItemResourceReleaseResult: ...
    def release_result_work_item_final_resources(
        self,
        host_dosage_buffer: object | None,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> NativeResultWorkItemResourceReleaseResult: ...
    def release_result_work_item_final_resources_for_object(
        self,
        work_item: object,
        has_released_host_dosage_buffer: bool,
    ) -> NativeResultWorkItemResourceReleaseResult: ...
    def release_result_work_item_in_flight_slot_for_object(
        self,
        work_item: object,
    ) -> NativeResultWorkItemResourceReleaseResult: ...
    def acquire_dosage_buffer_with_backpressure_timeout(self) -> NativeDosageBufferAcquireResult: ...
    def register_dosage_buffer(self, buffer_identifier: int) -> int: ...
    def register_dosage_buffer_with_observation(
        self,
        buffer_identifier: int,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def register_dosage_buffer_with_optional_observation(
        self,
        buffer_identifier: int,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def register_dosage_buffer_object_with_optional_observation(
        self,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def return_dosage_buffer(self, buffer_identifier: int, dosage_buffer: object) -> int | None: ...
    def return_dosage_buffer_with_observation(
        self,
        buffer_identifier: int,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def return_dosage_buffer_with_optional_observation(
        self,
        buffer_identifier: int,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def return_dosage_buffer_object_with_optional_observation(
        self,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def return_dosage_buffer_owner_with_optional_observation(
        self,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def release_numpy_dosage_buffer_with_optional_observation(
        self,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def discard_dosage_buffer(self, buffer_identifier: int) -> int | None: ...
    def discard_dosage_buffer_with_observation(
        self,
        buffer_identifier: int,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def discard_dosage_buffer_with_optional_observation(
        self,
        buffer_identifier: int,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def discard_dosage_buffer_object_with_optional_observation(
        self,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def discard_dosage_buffer_owner_with_optional_observation(
        self,
        dosage_buffer: object,
    ) -> NativeDosageBufferPoolOperationResult: ...
    def plan_dosage_buffer_return_attempt(self, buffer_identifier: int) -> NativeDosageBufferReturnAttemptPlan: ...
    def plan_dosage_buffer_object_return_attempt(
        self,
        dosage_buffer: object,
    ) -> NativeDosageBufferReturnAttemptPlan: ...
    def get_releasable_dosage_buffer_owner(
        self,
        dosage_buffer: object,
    ) -> object | None: ...
    def plan_dosage_buffer_reuse(
        self,
        buffered_shape: typing.Sequence[int],
        expected_shape: typing.Sequence[int],
    ) -> NativeDosageBufferReusePlan | None: ...
    def get_reusable_dosage_buffer(
        self,
        dosage_buffer: object,
        expected_shape: typing.Sequence[int],
        expected_dtype: object,
    ) -> object | None: ...
    def select_reusable_dosage_buffer_or_discard(
        self,
        dosage_buffer: object,
        expected_shape: typing.Sequence[int],
        expected_dtype: object,
    ) -> NativeDosageBufferReuseSelectionResult: ...
    def try_put_dosage_work_item(self, work_item: object, timeout_seconds: float) -> bool: ...
    def try_put_dosage_work_item_with_backpressure_timeout(self, work_item: object) -> bool: ...
    def put_dosage_work_item_with_backpressure_observation(
        self,
        work_item: object,
    ) -> NativeCallbackQueuePutObservationPlan: ...
    def put_dosage_work_item_with_optional_backpressure_observation(
        self,
        work_item: object,
    ) -> NativeCallbackQueuePutResult: ...
    def get_dosage_work_item(self) -> NativeCallbackObjectQueueGetResult: ...
    def get_dosage_work_item_with_observation(self) -> NativeCallbackQueueGetObservedResult: ...
    def get_dosage_work_item_with_drain_completion(self) -> NativeDosageWorkItemDrainResult: ...
    def get_dosage_work_item_with_observation_and_drain_completion(self) -> NativeDosageWorkItemGetResult: ...
    def get_dosage_work_item_with_optional_observation_and_drain_completion(
        self,
    ) -> NativeDosageWorkItemGetResult: ...
    def get_validated_dosage_work_item_with_drain_completion(
        self,
    ) -> NativeDosageWorkItemGetResult: ...
    def get_validated_dosage_work_item_with_optional_observation_and_drain_completion(
        self,
    ) -> NativeDosageWorkItemGetResult: ...
    def plan_dosage_work_drain_completion(
        self,
        has_dosage_work_item: bool,
    ) -> NativeDosageWorkDrainCompletionPlan: ...
    def plan_dosage_work_drain_completion_for_object(
        self,
        work_item: object,
    ) -> NativeDosageWorkDrainCompletionPlan: ...
    def plan_validated_dosage_work_item_dispatch(
        self,
        dosage_work_item_kind: str,
    ) -> NativeDosageWorkItemDispatchPlan: ...
    def plan_validated_dosage_work_item_dispatch_for_object(
        self,
        work_item: object,
    ) -> NativeDosageWorkItemDispatchPlan: ...
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
    def plan_dosage_queue_put_observation(self, queued: bool) -> NativeCallbackQueuePutObservationPlan: ...
    def plan_dosage_queue_get_observation(self) -> NativeCallbackQueueGetObservationPlan: ...
    def plan_result_queue_put_observation(self, queued: bool) -> NativeCallbackQueuePutObservationPlan: ...
    def plan_result_queue_get_observation(self) -> NativeCallbackQueueGetObservationPlan: ...
    def plan_dosage_buffer_pool_reuse_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_return_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_allocate_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_discard_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_consumer_wait_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
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
    def try_put_result_write_item(self, work_item: object, timeout_seconds: float) -> bool: ...
    def try_put_result_write_item_with_backpressure_timeout(self, work_item: object) -> bool: ...
    def put_result_write_item_with_backpressure_observation(
        self,
        work_item: object,
    ) -> NativeCallbackQueuePutObservationPlan: ...
    def put_result_write_item_with_optional_backpressure_observation(
        self,
        work_item: object,
    ) -> NativeCallbackQueuePutResult: ...
    def get_result_write_item(self) -> NativeCallbackObjectQueueGetResult: ...
    def get_result_write_item_with_observation(self) -> NativeCallbackQueueGetObservedResult: ...
    def get_result_write_item_with_drain_completion(self) -> NativeResultWriteItemDrainResult: ...
    def get_result_write_item_with_observation_and_drain_completion(self) -> NativeResultWriteItemGetResult: ...
    def get_result_write_item_with_optional_observation_and_drain_completion(
        self,
    ) -> NativeResultWriteItemGetResult: ...
    def get_validated_result_write_item_with_drain_completion(
        self,
    ) -> NativeResultWriteItemGetResult: ...
    def get_validated_result_write_item_with_optional_observation_and_drain_completion(
        self,
    ) -> NativeResultWriteItemGetResult: ...
    def plan_result_write_drain_completion(
        self,
        has_result_work_item: bool,
    ) -> NativeResultWriteDrainCompletionPlan: ...
    def plan_result_write_drain_completion_for_object(
        self,
        work_item: object,
    ) -> NativeResultWriteDrainCompletionPlan: ...
    def plan_validated_result_write_item_dispatch(
        self,
        result_work_item_kind: str,
    ) -> NativeResultWriteItemDispatchPlan: ...
    def plan_validated_result_write_item_dispatch_for_object(
        self,
        work_item: object,
    ) -> NativeResultWriteItemDispatchPlan: ...

class NativeCallbackQueueOperationObservationPlan:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def blocked_seconds(self) -> float: ...

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

class NativeCallbackQueueStageObservationPlan:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def stage_name(self) -> str: ...
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

class NativeCallbackQueuePutAttemptPlan:
    @property
    def should_put(self) -> bool: ...
    @property
    def should_wait(self) -> bool: ...
    @property
    def wait_timeout_seconds(self) -> float: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def queue_capacity(self) -> int: ...

class NativeCallbackQueuePutObservationPlan:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def blocked(self) -> bool: ...
    @property
    def should_retry_put(self) -> bool: ...

class NativeCallbackQueuePutResult:
    @property
    def should_retry_put(self) -> bool: ...
    @property
    def observation_plan(self) -> NativeCallbackQueuePutObservationPlan | None: ...
    @property
    def stage_backpressure_observation(self) -> NativeCallbackQueueStageBackpressureObservation | None: ...

class NativeCallbackQueueGetAttemptPlan:
    @property
    def should_get(self) -> bool: ...
    @property
    def should_wait(self) -> bool: ...
    @property
    def has_release_error(self) -> bool: ...
    @property
    def wait_timeout_seconds(self) -> float: ...
    @property
    def queue_depth(self) -> int: ...
    @property
    def queue_capacity(self) -> int: ...

class NativeCallbackQueueGetObservationPlan:
    @property
    def queue_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def blocked(self) -> bool: ...

class NativeResultInFlightAcquireAttemptPlan:
    @property
    def should_acquire(self) -> bool: ...
    @property
    def should_wait(self) -> bool: ...
    @property
    def wait_timeout_seconds(self) -> float: ...
    @property
    def occupied_count(self) -> int: ...
    @property
    def slot_limit(self) -> int: ...

class NativeResultInFlightAcquireObservationPlan:
    @property
    def resource_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def blocked(self) -> bool: ...
    @property
    def should_retry_acquisition(self) -> bool: ...

class NativeResultInFlightAcquireResult:
    @property
    def should_retry_acquisition(self) -> bool: ...
    @property
    def observation_plan(self) -> NativeResultInFlightAcquireObservationPlan | None: ...
    @property
    def stage_backpressure_observation(self) -> NativeCallbackQueueStageBackpressureObservation | None: ...

class NativeResultInFlightSlotReleaseResult:
    @property
    def observation_plan(self) -> NativeResultInFlightReleaseObservationPlan | None: ...
    @property
    def backpressure_observation(self) -> NativeCallbackQueueBackpressureObservation | None: ...

class NativeResultInFlightReleaseAttemptPlan:
    @property
    def should_release(self) -> bool: ...
    @property
    def has_release_error(self) -> bool: ...
    @property
    def occupied_count(self) -> int: ...
    @property
    def slot_limit(self) -> int: ...

class NativeResultInFlightReleaseObservationPlan:
    @property
    def resource_name(self) -> str: ...
    @property
    def operation_name(self) -> str: ...
    @property
    def blocked(self) -> bool: ...

class NativeResultWriteItemResourceReleasePlan:
    @property
    def should_release_host_buffer(self) -> bool: ...
    @property
    def should_release_result_in_flight_slot(self) -> bool: ...

class NativeResultWriteHandoffPlan:
    @property
    def should_enqueue(self) -> bool: ...
    @property
    def has_result_work_item(self) -> bool: ...
    @property
    def is_stop_signal(self) -> bool: ...

class NativeResultWriteDrainCompletionPlan:
    @property
    def should_stop(self) -> bool: ...
    @property
    def should_flush_binary_correction_diagnostics(self) -> bool: ...

class NativeResultWriteItemDispatchPlan:
    @property
    def result_work_item_kind(self) -> str: ...
    @property
    def expected_result_work_item_kind(self) -> str: ...
    @property
    def should_process_result_write_item(self) -> bool: ...
    @property
    def should_process_multi_result_write_item(self) -> bool: ...
    @property
    def has_dispatch_error(self) -> bool: ...
    @property
    def error_message(self) -> str | None: ...

class NativeDosageWorkDrainCompletionPlan:
    @property
    def should_stop(self) -> bool: ...

class NativeDosageWorkItemDispatchPlan:
    @property
    def dosage_work_item_kind(self) -> str: ...
    @property
    def should_process_sample_major_dosage(self) -> bool: ...
    @property
    def should_process_variant_major_dosage(self) -> bool: ...
    @property
    def should_process_variant_major_dosage_batch(self) -> bool: ...
    @property
    def should_process_variant_major_packed8_probability_pair(self) -> bool: ...
    @property
    def has_dispatch_error(self) -> bool: ...
    @property
    def error_message(self) -> str | None: ...

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

class NativeDosageBufferAcquireAttemptPlan:
    @property
    def should_take_free_buffer(self) -> bool: ...
    @property
    def should_allocate(self) -> bool: ...
    @property
    def should_wait(self) -> bool: ...
    @property
    def wait_timeout_seconds(self) -> float: ...
    @property
    def free_buffer_count(self) -> int: ...
    @property
    def allocated_count(self) -> int: ...
    @property
    def buffer_limit(self) -> int: ...

class NativeDosageBufferRegisterAttemptPlan:
    @property
    def should_register(self) -> bool: ...
    @property
    def has_registration_error(self) -> bool: ...
    @property
    def allocated_count(self) -> int: ...
    @property
    def buffer_limit(self) -> int: ...

class NativeDosageBufferReturnAttemptPlan:
    @property
    def should_return(self) -> bool: ...
    @property
    def allocated_count(self) -> int: ...
    @property
    def buffer_limit(self) -> int: ...

class NativeDosageBufferDiscardAttemptPlan:
    @property
    def should_discard(self) -> bool: ...
    @property
    def allocated_count(self) -> int: ...
    @property
    def buffer_limit(self) -> int: ...

class NativeDosageBufferPoolObservationPlan:
    @property
    def operation_name(self) -> str: ...
    @property
    def blocked(self) -> bool: ...

class NativeCallbackSchedulerState:
    def __init__(
        self,
        staging_depth: int,
        native_callback_batch_size: int,
        result_in_flight_limit: int | None,
        dosage_buffer_limit: int | None,
    ) -> None: ...
    @property
    def native_callback_batch_size(self) -> int: ...
    @property
    def dosage_queue_depth(self) -> int: ...
    @property
    def dosage_queue_capacity(self) -> int: ...
    @property
    def dosage_queue_occupied_count(self) -> int: ...
    def has_available_dosage_queue_slot(self) -> bool: ...
    def acquire_dosage_queue_slot(self) -> bool: ...
    def release_dosage_queue_slot(self) -> bool: ...
    def plan_dosage_queue_put_attempt(self, wait_timeout_seconds: float) -> NativeCallbackQueuePutAttemptPlan: ...
    def plan_dosage_queue_put_backpressure_attempt(self) -> NativeCallbackQueuePutAttemptPlan: ...
    def plan_dosage_queue_put_observation(self, queued: bool) -> NativeCallbackQueuePutObservationPlan: ...
    def plan_dosage_queue_get_attempt(self, has_queued_item: bool) -> NativeCallbackQueueGetAttemptPlan: ...
    def plan_dosage_queue_get_observation(self) -> NativeCallbackQueueGetObservationPlan: ...
    @property
    def result_queue_depth(self) -> int: ...
    @property
    def result_queue_capacity(self) -> int: ...
    @property
    def result_queue_occupied_count(self) -> int: ...
    def has_available_result_queue_slot(self) -> bool: ...
    def acquire_result_queue_slot(self) -> bool: ...
    def release_result_queue_slot(self) -> bool: ...
    def plan_result_queue_put_attempt(self, wait_timeout_seconds: float) -> NativeCallbackQueuePutAttemptPlan: ...
    def plan_result_queue_put_backpressure_attempt(self) -> NativeCallbackQueuePutAttemptPlan: ...
    def plan_result_queue_put_observation(self, queued: bool) -> NativeCallbackQueuePutObservationPlan: ...
    def plan_result_queue_get_attempt(self, has_queued_item: bool) -> NativeCallbackQueueGetAttemptPlan: ...
    def plan_result_queue_get_observation(self) -> NativeCallbackQueueGetObservationPlan: ...
    @property
    def result_in_flight_limit(self) -> int: ...
    @property
    def dosage_buffer_limit(self) -> int: ...
    @property
    def has_started(self) -> bool: ...
    def mark_started(self) -> bool: ...
    def plan_worker_start(self) -> NativeCallbackWorkerStartPlan: ...
    def plan_worker_start_attempt(self) -> NativeCallbackWorkerStartAttemptPlan: ...
    @property
    def result_in_flight_slot_limit(self) -> int: ...
    @property
    def result_in_flight_occupied_count(self) -> int: ...
    def has_available_result_in_flight_slot(self) -> bool: ...
    def acquire_result_in_flight_slot(self) -> bool: ...
    def release_result_in_flight_slot(self) -> bool: ...
    def plan_result_in_flight_slot_acquire_attempt(
        self,
        wait_timeout_seconds: float,
    ) -> NativeResultInFlightAcquireAttemptPlan: ...
    def plan_result_in_flight_slot_acquire_backpressure_attempt(self) -> NativeResultInFlightAcquireAttemptPlan: ...
    def plan_result_in_flight_slot_acquire_observation(
        self,
        acquire_attempt_plan: NativeResultInFlightAcquireAttemptPlan,
    ) -> NativeResultInFlightAcquireObservationPlan: ...
    def plan_result_in_flight_slot_release_attempt(self) -> NativeResultInFlightReleaseAttemptPlan: ...
    def plan_result_in_flight_slot_release_observation(self) -> NativeResultInFlightReleaseObservationPlan: ...
    def plan_result_write_item_pre_write_resource_release(
        self,
        has_host_dosage_buffer: bool,
    ) -> NativeResultWriteItemResourceReleasePlan: ...
    def plan_result_write_item_final_resource_release(
        self,
        has_host_dosage_buffer: bool,
        has_released_host_dosage_buffer: bool,
        release_in_flight_slot: bool,
    ) -> NativeResultWriteItemResourceReleasePlan: ...
    def plan_result_write_handoff(self, has_result_work_item: bool) -> NativeResultWriteHandoffPlan: ...
    def plan_result_write_drain_completion(
        self,
        has_result_work_item: bool,
        flush_binary_correction_diagnostics_on_stop: bool,
    ) -> NativeResultWriteDrainCompletionPlan: ...
    def plan_result_write_item_dispatch(
        self,
        result_work_item_kind: str,
        expected_result_work_item_kind: str,
    ) -> NativeResultWriteItemDispatchPlan: ...
    def plan_dosage_work_drain_completion(
        self,
        has_dosage_work_item: bool,
    ) -> NativeDosageWorkDrainCompletionPlan: ...
    def plan_dosage_work_item_dispatch(
        self,
        dosage_work_item_kind: str,
    ) -> NativeDosageWorkItemDispatchPlan: ...
    def plan_dosage_work_item_stage_duration(
        self,
        dosage_work_item_kind: str,
        chunk_count: int,
        elapsed_seconds: float,
    ) -> NativeDosageWorkItemStageDurationPlan: ...
    @property
    def dosage_buffer_pool_limit(self) -> int: ...
    @property
    def dosage_buffer_allocated_count(self) -> int: ...
    @property
    def dosage_buffer_identifiers(self) -> list[int]: ...
    def has_available_dosage_buffer_slot(self) -> bool: ...
    def owns_dosage_buffer(self, buffer_identifier: int) -> bool: ...
    def register_dosage_buffer(self, buffer_identifier: int) -> bool: ...
    def discard_dosage_buffer(self, buffer_identifier: int) -> bool: ...
    def plan_dosage_buffer_acquire_attempt(
        self,
        free_buffer_count: int,
        wait_timeout_seconds: float,
    ) -> NativeDosageBufferAcquireAttemptPlan: ...
    def plan_dosage_buffer_acquire_backpressure_attempt(
        self,
        free_buffer_count: int,
    ) -> NativeDosageBufferAcquireAttemptPlan: ...
    def plan_dosage_buffer_register_attempt(self, buffer_identifier: int) -> NativeDosageBufferRegisterAttemptPlan: ...
    def plan_dosage_buffer_return_attempt(self, buffer_identifier: int) -> NativeDosageBufferReturnAttemptPlan: ...
    def plan_dosage_buffer_discard_attempt(self, buffer_identifier: int) -> NativeDosageBufferDiscardAttemptPlan: ...
    def plan_dosage_buffer_pool_reuse_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_return_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_allocate_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_discard_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_pool_consumer_wait_observation(self) -> NativeDosageBufferPoolObservationPlan: ...
    def plan_dosage_buffer_reuse(
        self,
        buffered_shape: typing.Sequence[int],
        expected_shape: typing.Sequence[int],
    ) -> NativeDosageBufferReusePlan | None: ...
    def plan_variant_major_dosage_batch_handoff(
        self,
        metadata_count: int,
        genotype_matrix_by_variant_count: int,
        chunk_stats_count: int,
    ) -> NativeVariantMajorDosageBatchHandoffPlan: ...
    def plan_dosage_work_handoff(self, chunk_count: int) -> NativeDosageWorkHandoffPlan: ...
    @property
    def dosage_worker_error_message(self) -> str | None: ...
    @property
    def result_worker_error_message(self) -> str | None: ...
    @property
    def has_dosage_worker_error(self) -> bool: ...
    @property
    def has_result_worker_error(self) -> bool: ...
    def record_dosage_worker_error(self, error_message: str) -> None: ...
    def record_result_worker_error(self, error_message: str) -> None: ...
    def update_dosage_worker_error(self, error_message: str | None) -> NativeCallbackWorkerErrorUpdatePlan: ...
    def update_result_worker_error(self, error_message: str | None) -> NativeCallbackWorkerErrorUpdatePlan: ...
    def clear_dosage_worker_error(self) -> bool: ...
    def clear_result_worker_error(self) -> bool: ...
    @property
    def backpressure_poll_timeout_seconds(self) -> float: ...
    def plan_worker_finish(self) -> NativeCallbackWorkerFinishPlan: ...
    def plan_worker_abort(self) -> NativeCallbackWorkerAbortPlan: ...
    def plan_worker_error_raise(self) -> NativeCallbackWorkerErrorRaisePlan: ...
    def plan_queue_operation_observation(
        self,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueOperationObservationPlan: ...
    def plan_queue_backpressure_observation(
        self,
        queue_name: str,
        operation_name: str,
        queue_depth: int,
        queue_capacity: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueBackpressureObservation: ...
    def plan_current_queue_backpressure_observation(
        self,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueBackpressureObservation: ...
    def plan_dosage_buffer_pool_backpressure_observation(
        self,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueBackpressureObservation: ...
    def plan_queue_stage_observation(
        self,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueStageObservationPlan: ...
    def plan_queue_stage_backpressure_observation(
        self,
        queue_name: str,
        operation_name: str,
        queue_depth: int,
        queue_capacity: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueStageBackpressureObservation: ...
    def plan_current_queue_stage_backpressure_observation(
        self,
        queue_name: str,
        operation_name: str,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueStageBackpressureObservation: ...
    def plan_dosage_buffer_pool_stage_backpressure_observation(
        self,
        operation_name: str,
        free_buffer_count: int,
        elapsed_seconds: float,
        blocked: bool,
    ) -> NativeCallbackQueueStageBackpressureObservation: ...
    def plan_dosage_worker_join(self, timeout_seconds: float | None) -> NativeCallbackWorkerJoinPlan: ...
    def plan_result_worker_join(self, timeout_seconds: float | None) -> NativeCallbackWorkerJoinPlan: ...
    def plan_dosage_worker_stop(
        self,
        timeout_seconds: float | None,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPlan: ...
    def plan_result_worker_stop(
        self,
        timeout_seconds: float | None,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPlan: ...
    def plan_dosage_worker_stop_poll(
        self,
        remaining_timeout_seconds: float,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPollPlan: ...
    def plan_result_worker_stop_poll(
        self,
        remaining_timeout_seconds: float,
        is_worker_alive: bool,
    ) -> NativeCallbackWorkerStopPollPlan: ...

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

def emit_callback_progress_update_telemetry(
    telemetry_session: object | None,
    progress_update: NativeCallbackProgressUpdate | None,
) -> None: ...
def emit_callback_progress_event_telemetry(
    telemetry_session: object | None,
    progress_event: NativeCallbackProgressTelemetryEvent | None,
    missing_session_message: str,
) -> None: ...
def emit_callback_progress_completion_telemetry(
    telemetry_session: object | None,
    progress_completion: NativeCallbackProgressCompletion | None,
) -> None: ...

class NativeDosageBufferReusePlan:
    @property
    def requires_slice(self) -> bool: ...
    @property
    def slice_dimensions(self) -> list[int]: ...

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
    def stage_timing_json_payload(self) -> dict[str, object]: ...
    def write_stage_timing_snapshot(self, path: str) -> None: ...
    def write_stage_timing_snapshot_if_configured(self, path: str | None) -> bool: ...
    def derived_metrics_payload(self) -> dict[str, float]: ...
    def profile_summary_payload(self, run_id: str | None) -> dict[str, object]: ...
    def write_profile_summary(self, path: str, run_id: str | None) -> None: ...
    def write_profile_summary_if_configured(self, path: str | None, run_id: str | None) -> bool: ...
    def write_final_timing_outputs(
        self,
        stage_timing_path: str | None,
        profile_summary_path: str | None,
        run_id: str | None,
    ) -> dict[str, bool]: ...

class NativeStageTimingRecorderPlan:
    should_create: bool
    exact_stage_timings: bool

class NativeTimingFileWritePlan:
    should_write: bool

class NativeFinalTimingOutputContext:
    stage_timing_path: str | None
    profile_summary_path: str | None
    run_id: str | None
    force_stage_timing_recorder: bool

class NativeCliRunFailureTelemetryPlan:
    should_log_run_failed_to_telemetry: bool

class NativeCliTelemetryCloseFailurePlan:
    should_report_failure: bool
    exit_code: int

class NativeCliRunLifecycleState:
    def __init__(self) -> None: ...
    @property
    def runner_started(self) -> bool: ...
    def mark_runner_started(self) -> None: ...
    def plan_run_failed_telemetry(self) -> NativeCliRunFailureTelemetryPlan: ...

def emit_cli_run_failed_telemetry_event(
    telemetry_session: object | None,
    failed_event: object,
    should_log_run_failed_to_telemetry: bool,
) -> None: ...
def plan_cli_telemetry_close_failure(
    current_exit_code: int,
    runtime_failure_exit_code: int,
) -> NativeCliTelemetryCloseFailurePlan: ...
def resolve_final_timing_output_context(
    diagnostics_stage_timing_path: str | None,
    telemetry_session: object | None,
) -> NativeFinalTimingOutputContext: ...

class NativeRuntimeCompatibilityToken:
    pass

class NativeRuntimePolicy:
    rayon_thread_count: int | None
    def logging_runtime_policy_payload(self) -> dict[str, object]: ...
    def jax_runtime_policy_payload(self) -> dict[str, object]: ...

class NativeRunRuntime:
    rayon_thread_count: int | None
    def logging_runtime_policy_payload(self) -> dict[str, object]: ...
    def jax_runtime_policy_payload(self) -> dict[str, object]: ...
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

class NativeJaxRuntimeSetupSession:
    def __init__(self, setup_payload: dict[str, object], should_configure: bool) -> None: ...
    @property
    def should_configure(self) -> bool: ...
    def setup_payload(self) -> dict[str, object]: ...
    def side_effect_plan_payload(self) -> dict[str, object]: ...
    def config_update_payloads(self) -> tuple[dict[str, object], ...]: ...
    def apply_config_updates(self) -> int: ...
    def complete_validation_payload(
        self,
        gpu_validation_status: str,
        gpu_validation_message: str | None,
    ) -> dict[str, object]: ...
    def diagnostic_event_payloads(self) -> tuple[dict[str, object], ...]: ...
    def create_cache_directory_if_configured(self) -> bool: ...
    def validate_gpu_if_configured(
        self,
        control_device_path: str,
        uvm_device_path: str,
        driver_directory_path: str,
    ) -> dict[str, object]: ...
    def validate_gpu_if_configured_with_default_probe_paths(self) -> dict[str, object]: ...

class NativeRuntimeState:
    rayon_thread_count: int | None
    def __init__(self) -> None: ...
    def logging_runtime_policy_payload(self) -> dict[str, object] | None: ...
    def jax_runtime_policy_payload(self) -> dict[str, object] | None: ...
    def runtime_state_payload(self) -> dict[str, object]: ...
    def require_compatible_runtime_policy(
        self,
        logging_policy_payload: dict[str, object],
        rayon_thread_count: int | None,
        jax_policy_payload: dict[str, object],
    ) -> NativeRuntimeCompatibilityToken: ...
    def build_run_runtime(
        self,
        runtime_policy: NativeRuntimePolicy,
    ) -> NativeRunRuntime: ...
    def require_compatible_runtime_policy_handle(
        self,
        runtime_policy: NativeRuntimePolicy,
    ) -> NativeRuntimeCompatibilityToken: ...
    def require_compatible_logging_runtime_policy(self, payload: dict[str, object]) -> None: ...
    def record_logging_runtime_policy(self, payload: dict[str, object]) -> None: ...
    def initialize_logging_runtime_policy(self, payload: dict[str, object]) -> bool: ...
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
    def require_compatible_jax_runtime_policy(self, payload: dict[str, object]) -> None: ...
    def record_jax_runtime_policy(self, payload: dict[str, object]) -> None: ...
    def complete_jax_runtime_setup(self, payload: dict[str, object]) -> None: ...
    def complete_jax_runtime_setup_session(
        self,
        payload: dict[str, object],
        setup_session: NativeJaxRuntimeSetupSession,
    ) -> None: ...
    def plan_jax_runtime_setup_lifecycle(
        self,
        payload: dict[str, object],
    ) -> NativeJaxRuntimeSetupLifecyclePlan: ...
    def build_jax_runtime_setup_session(
        self,
        payload: dict[str, object],
        resolved_cache_directory: str,
    ) -> NativeJaxRuntimeSetupSession: ...

def global_process_runtime_state() -> NativeRuntimeState: ...
def build_process_runtime_state_handle(
    logging_policy_payload: dict[str, object] | None,
    rayon_thread_count: int | None,
    jax_policy_payload: dict[str, object] | None,
) -> NativeRuntimeState: ...

class NativeSecondSignalExceptionPlan:
    raise_keyboard_interrupt: bool
    exit_code: int

class NativeTrustedBgenValidationCacheLookupPlan:
    @property
    def should_mark_validated(self) -> bool: ...
    @property
    def should_validate(self) -> bool: ...
    @property
    def should_write_cache(self) -> bool: ...

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
    @property
    def existing_manifest_json(self) -> str | None: ...

class NativeInitializedOutputRun:
    @property
    def committed_chunk_identifiers(self) -> list[int]: ...

class NativeManifestFileFingerprintCache:
    def __init__(self) -> None: ...
    def build_file_fingerprint_payload(
        self,
        path: str,
        include_content_hash: bool,
    ) -> dict[str, object]: ...

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
    runtime_compatibility_token: NativeRuntimeCompatibilityToken,
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
def build_prepared_run_manifest_header_json_from_current_header_json(current_header_json: str) -> str: ...
def build_prepared_run_plan_json(prepared_run_plan_input_json: str) -> str: ...
def build_prepared_run_plan_json_from_current_header_json(current_header_json: str) -> str: ...
def build_prediction_loco_file_fingerprints_json(
    prediction_list_path: str,
    phenotype_names: list[str],
) -> str: ...
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
    runtime_compatibility_token: NativeRuntimeCompatibilityToken,
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
def build_runtime_policy_handle(
    logging_policy_payload: dict[str, object],
    rayon_thread_count: int | None,
    jax_policy_payload: dict[str, object],
) -> NativeRuntimePolicy: ...
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
def default_shutdown_signal_numbers() -> list[int]: ...
def plan_second_signal_exception(signal_number: int) -> NativeSecondSignalExceptionPlan: ...
def raise_second_signal_exception(signal_number: int) -> typing.NoReturn: ...
def configure_bgen_decode_tile_variant_count(tile_variant_count: int) -> None: ...
def configure_rayon_global_thread_pool(thread_count: int) -> None: ...
def format_rayon_thread_pool_configuration_error_value(thread_count: int, source_error: str) -> str: ...
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
def build_current_telemetry_event_payload(
    run_id: str,
    event: str,
    level: str,
    fields: dict[str, object],
) -> dict[str, object]: ...
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
def resolve_telemetry_session_policy_payload(
    telemetry_mode: str,
    trace_event_cap: int,
) -> dict[str, object]: ...
def paths_refer_to_same_file_value(first_path: str, second_path: str) -> bool: ...
def build_empty_telemetry_writer_counters_payload() -> dict[str, object]: ...
def generate_telemetry_run_id_value() -> str: ...
def attach_run_metadata_payload(
    artifacts: object,
    run_id: str | None,
    association_mode: str,
    phenotype_count: int,
) -> dict[str, object]: ...
def build_run_completed_event_payload(artifacts: object) -> dict[str, object]: ...
def build_run_interrupted_event_payload(shutdown_request: object) -> dict[str, object]: ...
def build_run_failed_event_payload(error: BaseException) -> dict[str, object]: ...
def build_run_completed_telemetry_fields(event: object) -> dict[str, object]: ...
def build_run_interrupted_telemetry_fields(event: object) -> dict[str, object]: ...
def build_run_failed_telemetry_fields(event: object) -> dict[str, object]: ...
def record_runner_run_started_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    trait_type: str,
    phenotype_count: int,
    output_run_root: str,
) -> None: ...
def record_runner_run_interrupted_telemetry_event(
    telemetry_session: object | None,
    event: object,
) -> None: ...
def record_runner_run_failed_telemetry_event(
    telemetry_session: object | None,
    event: object,
) -> None: ...
def record_runner_run_completed_telemetry_event(
    telemetry_session: object | None,
    event: object,
) -> None: ...
def record_execution_plan_prepared_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    trait_type: str,
    phenotype_count: int,
    chunk_size: int,
    variant_limit: int | None,
    device: str,
) -> None: ...
def record_effective_config_written_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    phenotype: str,
    effective_config: str,
    output_run_directory: str,
) -> None: ...
def record_writer_finished_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    phenotype: str,
    final_output_path: str | None,
) -> None: ...
def record_multi_writer_finished_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    phenotype_count: int,
    final_output_paths: typing.Sequence[str | None],
) -> None: ...
def record_single_trait_preflight_completed_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    phenotype: str,
    sample_count: int,
    covariate_count: int,
    chromosome_count: int,
) -> None: ...
def record_multi_phenotype_preflight_completed_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    phenotype_count: int,
    sample_count: int,
) -> None: ...
def record_sample_alignment_completed_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    phenotype: str | None,
    phenotype_count: int | None,
    sample_count: int | None,
    covariate_count: int | None,
    phenotype_group_count: int | None,
) -> None: ...
def record_prediction_source_loaded_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    phenotype: str | None,
    phenotype_count: int | None,
) -> None: ...
def record_multi_phenotype_sample_summary_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    sample_mode: str,
    sample_counts: typing.Sequence[int],
    sample_set_fingerprints: typing.Sequence[str | None],
    phenotype_group_count: int,
) -> None: ...
def record_gpu_genotype_format_resolved_telemetry_event(
    telemetry_session: object | None,
    requested_gpu_genotype_format: str,
    resolved_gpu_genotype_format: str,
    resolution_reason: str,
    fallback_error: str | None,
) -> None: ...
def record_association_backend_selected_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    association_backend_kind: str,
    device: str,
    genotype_format: str,
    phenotype: str | None,
    phenotype_count: int | None,
) -> None: ...
def record_bgen_engine_opened_telemetry_event(
    telemetry_session: object | None,
    association_mode: str,
    association_backend_kind: str,
    sample_count: int,
    variant_count: int,
    phenotype: str | None,
    phenotype_count: int | None,
) -> None: ...
def build_native_runtime_knobs_configured_diagnostic_payload(
    bgen_decode_tile_variant_count: int,
    threads: int | None,
) -> dict[str, object]: ...
def record_native_runtime_knobs_configured_diagnostic_event(
    bgen_decode_tile_variant_count: int,
    threads: int | None,
) -> None: ...
def build_runner_metadata_artifacts_finalized_diagnostic_payload(
    association_mode: str,
    phenotype_count: int,
) -> dict[str, object]: ...
def record_runner_metadata_artifacts_finalized_diagnostic_event(
    association_mode: str,
    phenotype_count: int,
) -> None: ...
def build_preflight_warning_diagnostic_payload(
    message: str,
    chromosome_count: int,
    covariate_count: int,
    preflight_scope: str,
    sample_count: int,
    trusted_no_missing_diploid: bool,
    warning_index: int,
) -> dict[str, object]: ...
def record_preflight_warning_diagnostic_event(
    message: str,
    chromosome_count: int,
    covariate_count: int,
    preflight_scope: str,
    sample_count: int,
    trusted_no_missing_diploid: bool,
    warning_index: int,
) -> None: ...
def build_io_output_resume_committed_chunks_diagnostic_payload(
    chunks_directory: str,
    committed_chunk_count: int,
    run_directory: str,
) -> dict[str, object]: ...
def record_io_output_resume_committed_chunks_diagnostic_event(
    chunks_directory: str,
    committed_chunk_count: int,
    run_directory: str,
) -> None: ...
def build_pipeline_bgen_engine_open_started_diagnostic_payload(
    phenotype_count: int | None,
    phenotype_name: str | None,
    pipeline_label: str,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> dict[str, object]: ...
def record_pipeline_bgen_engine_open_started_diagnostic_event(
    phenotype_count: int | None,
    phenotype_name: str | None,
    pipeline_label: str,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> None: ...
def build_pipeline_bgen_engine_opened_diagnostic_payload(
    phenotype_count: int | None,
    phenotype_name: str | None,
    pipeline_label: str,
    sample_count: int,
    variant_count: int,
) -> dict[str, object]: ...
def record_pipeline_bgen_engine_opened_diagnostic_event(
    phenotype_count: int | None,
    phenotype_name: str | None,
    pipeline_label: str,
    sample_count: int,
    variant_count: int,
) -> None: ...
def build_pipeline_prevalidated_bgen_engine_used_diagnostic_payload(
    phenotype_count: int | None,
    phenotype_name: str | None,
    pipeline_label: str,
) -> dict[str, object]: ...
def record_pipeline_prevalidated_bgen_engine_used_diagnostic_event(
    phenotype_count: int | None,
    phenotype_name: str | None,
    pipeline_label: str,
) -> None: ...
def build_pipeline_output_resume_committed_chunks_diagnostic_payload(
    committed_chunk_count: int,
    output_index: int,
) -> dict[str, object]: ...
def record_pipeline_output_resume_committed_chunks_diagnostic_event(
    committed_chunk_count: int,
    output_index: int,
) -> None: ...
def build_pipeline_output_writer_sessions_create_started_diagnostic_payload(
    association_mode: str,
    output_count: int,
) -> dict[str, object]: ...
def record_pipeline_output_writer_sessions_create_started_diagnostic_event(
    association_mode: str,
    output_count: int,
) -> None: ...
def build_pipeline_gpu_genotype_format_resolved_diagnostic_payload(
    requested_gpu_genotype_format: str,
    resolved_gpu_genotype_format: str,
    resolution_reason: str,
    fallback_error: str | None,
) -> dict[str, object]: ...
def record_pipeline_gpu_genotype_format_resolved_diagnostic_event(
    requested_gpu_genotype_format: str,
    resolved_gpu_genotype_format: str,
    resolution_reason: str,
    fallback_error: str | None,
) -> None: ...
def build_callback_null_logistic_nonconvergence_warning_diagnostic_payload(
    message: str,
    chromosome: str,
    nonconverged_count: int,
    phenotype_count: int,
    policy: str,
    scalar_convergence: bool,
    total_fit_count: int,
) -> dict[str, object]: ...
def record_callback_null_logistic_nonconvergence_warning_diagnostic_event(
    message: str,
    chromosome: str,
    nonconverged_count: int,
    phenotype_count: int,
    policy: str,
    scalar_convergence: bool,
    total_fit_count: int,
) -> None: ...
def build_pipeline_multi_phenotype_sample_summary_diagnostic_payload(
    phenotype_count: int,
    phenotype_group_count: int,
    sample_counts_differ: bool,
    sample_mode: str,
) -> dict[str, object]: ...
def record_pipeline_multi_phenotype_sample_summary_diagnostic_event(
    phenotype_count: int,
    phenotype_group_count: int,
    sample_counts_differ: bool,
    sample_mode: str,
) -> None: ...
def build_pipeline_multi_trait_started_diagnostic_payload(
    association_mode: str,
    phenotype_count: int,
    sample_mode: str,
) -> dict[str, object]: ...
def record_pipeline_multi_trait_started_diagnostic_event(
    association_mode: str,
    phenotype_count: int,
    sample_mode: str,
) -> None: ...
def build_pipeline_multi_trait_input_load_started_diagnostic_payload(
    phenotype_count: int,
) -> dict[str, object]: ...
def record_pipeline_multi_trait_input_load_started_diagnostic_event(
    phenotype_count: int,
) -> None: ...
def build_pipeline_multi_trait_input_aligned_diagnostic_payload(
    covariate_count: int,
    phenotype_count: int,
    sample_count: int,
) -> dict[str, object]: ...
def record_pipeline_multi_trait_input_aligned_diagnostic_event(
    covariate_count: int,
    phenotype_count: int,
    sample_count: int,
) -> None: ...
def build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload(
    phenotype_count: int,
) -> dict[str, object]: ...
def record_pipeline_multi_trait_prediction_source_load_started_diagnostic_event(
    phenotype_count: int,
) -> None: ...
def build_pipeline_grouped_per_phenotype_started_diagnostic_payload(
    association_mode: str,
    phenotype_count: int,
    sample_mode: str,
) -> dict[str, object]: ...
def record_pipeline_grouped_per_phenotype_started_diagnostic_event(
    association_mode: str,
    phenotype_count: int,
    sample_mode: str,
) -> None: ...
def build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload(
    phenotype_count: int,
    phenotype_group_count: int,
) -> dict[str, object]: ...
def record_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_event(
    phenotype_count: int,
    phenotype_group_count: int,
) -> None: ...
def build_pipeline_grouped_union_delivery_selected_diagnostic_payload(
    grouped_sample_count: int,
    phenotype_group_count: int,
    union_sample_count: int,
) -> dict[str, object]: ...
def record_pipeline_grouped_union_delivery_selected_diagnostic_event(
    grouped_sample_count: int,
    phenotype_group_count: int,
    union_sample_count: int,
) -> None: ...
def build_pipeline_multi_group_preflight_started_diagnostic_payload(
    phenotype_count: int,
    sample_count: int,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> dict[str, object]: ...
def record_pipeline_multi_group_preflight_started_diagnostic_event(
    phenotype_count: int,
    sample_count: int,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> None: ...
def build_pipeline_multi_group_preflight_completed_diagnostic_payload(
    phenotype_count: int,
    sample_count: int,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> dict[str, object]: ...
def record_pipeline_multi_group_preflight_completed_diagnostic_event(
    phenotype_count: int,
    sample_count: int,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> None: ...
def build_pipeline_single_trait_started_diagnostic_payload(
    association_mode: str,
    phenotype_name: str,
    pipeline_label: str,
) -> dict[str, object]: ...
def record_pipeline_single_trait_started_diagnostic_event(
    association_mode: str,
    phenotype_name: str,
    pipeline_label: str,
) -> None: ...
def build_pipeline_single_trait_input_load_started_diagnostic_payload(
    phenotype_name: str,
    pipeline_label: str,
) -> dict[str, object]: ...
def record_pipeline_single_trait_input_load_started_diagnostic_event(
    phenotype_name: str,
    pipeline_label: str,
) -> None: ...
def build_pipeline_single_trait_input_aligned_diagnostic_payload(
    covariate_count: int,
    phenotype_name: str,
    pipeline_label: str,
    sample_count: int,
) -> dict[str, object]: ...
def record_pipeline_single_trait_input_aligned_diagnostic_event(
    covariate_count: int,
    phenotype_name: str,
    pipeline_label: str,
    sample_count: int,
) -> None: ...
def build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload(
    phenotype_name: str,
    pipeline_label: str,
) -> dict[str, object]: ...
def record_pipeline_single_trait_prediction_source_load_started_diagnostic_event(
    phenotype_name: str,
    pipeline_label: str,
) -> None: ...
def build_pipeline_single_trait_preflight_started_diagnostic_payload(
    phenotype_name: str,
    pipeline_label: str,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> dict[str, object]: ...
def record_pipeline_single_trait_preflight_started_diagnostic_event(
    phenotype_name: str,
    pipeline_label: str,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> None: ...
def build_pipeline_single_trait_preflight_completed_diagnostic_payload(
    chromosome_count: int,
    covariate_count: int,
    phenotype_name: str,
    pipeline_label: str,
    sample_count: int,
) -> dict[str, object]: ...
def record_pipeline_single_trait_preflight_completed_diagnostic_event(
    chromosome_count: int,
    covariate_count: int,
    phenotype_name: str,
    pipeline_label: str,
    sample_count: int,
) -> None: ...
def build_native_dispatch_bgen_engine_constructing_diagnostic_payload(
    chunk_size: int,
    source_path: str,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> dict[str, object]: ...
def record_native_dispatch_bgen_engine_constructing_diagnostic_event(
    chunk_size: int,
    source_path: str,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> None: ...
def build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload(
    source_path: str,
    trusted_bgen_validation_mode: str,
) -> dict[str, object]: ...
def record_native_dispatch_trusted_bgen_validation_started_diagnostic_event(
    source_path: str,
    trusted_bgen_validation_mode: str,
) -> None: ...
def build_native_dispatch_callback_drain_started_diagnostic_payload() -> dict[str, object]: ...
def build_native_dispatch_delivery_started_diagnostic_payload(
    committed_chunk_count: int,
    pipeline_label: str,
    variant_major_packed8_probability_pairs: bool,
) -> dict[str, object]: ...
def record_native_dispatch_delivery_started_diagnostic_event(
    committed_chunk_count: int,
    pipeline_label: str,
    variant_major_packed8_probability_pairs: bool,
) -> None: ...
def build_native_dispatch_delivery_finished_diagnostic_payload(
    pipeline_label: str,
    processed_chunk_count: int,
) -> dict[str, object]: ...
def record_native_dispatch_delivery_finished_diagnostic_event(
    pipeline_label: str,
    processed_chunk_count: int,
) -> None: ...
def build_native_dispatch_delivery_interrupted_diagnostic_payload(
    pipeline_label: str,
    signal_exit_code: int,
    signal_name: str,
    signal_number: int,
) -> dict[str, object]: ...
def record_native_dispatch_delivery_interrupted_diagnostic_event(
    pipeline_label: str,
    signal_exit_code: int,
    signal_name: str,
    signal_number: int,
) -> None: ...
def build_native_dispatch_delivery_failed_diagnostic_payload(
    exception_message: str,
    exception_type: str,
    pipeline_label: str,
) -> dict[str, object]: ...
def record_native_dispatch_delivery_failed_diagnostic_event(
    exception_message: str,
    exception_type: str,
    pipeline_label: str,
) -> None: ...
def build_native_dispatch_pipeline_finished_diagnostic_payload(
    final_parquet_path_count: int,
    pipeline_label: str,
) -> dict[str, object]: ...
def record_native_dispatch_pipeline_finished_diagnostic_event(
    final_parquet_path_count: int,
    pipeline_label: str,
) -> None: ...
def record_native_dispatch_callback_drain_started_diagnostic_event() -> None: ...
def build_native_dispatch_writer_session_finish_started_diagnostic_payload() -> dict[str, object]: ...
def record_native_dispatch_writer_session_finish_started_diagnostic_event() -> None: ...
def build_native_dispatch_writer_sessions_finish_started_diagnostic_payload(
    requested_thread_count: int,
    writer_session_count: int,
) -> dict[str, object]: ...
def record_native_dispatch_writer_sessions_finish_started_diagnostic_event(
    requested_thread_count: int,
    writer_session_count: int,
) -> None: ...
def build_native_dispatch_writer_session_interrupted_flush_started_diagnostic_payload(
    signal_exit_code: int,
    signal_name: str,
    signal_number: int,
) -> dict[str, object]: ...
def record_native_dispatch_writer_session_interrupted_flush_started_diagnostic_event(
    signal_exit_code: int,
    signal_name: str,
    signal_number: int,
) -> None: ...
def build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload(
    requested_thread_count: int,
    signal_exit_code: int,
    signal_name: str,
    signal_number: int,
    writer_session_count: int,
) -> dict[str, object]: ...
def record_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_event(
    requested_thread_count: int,
    signal_exit_code: int,
    signal_name: str,
    signal_number: int,
    writer_session_count: int,
) -> None: ...
def build_runner_run_started_diagnostic_payload(
    association_mode: str,
    trait_type: str,
    phenotype_count: int,
) -> dict[str, object]: ...
def record_runner_run_started_diagnostic_event(
    association_mode: str,
    trait_type: str,
    phenotype_count: int,
) -> None: ...
def build_runner_run_interrupted_diagnostic_payload(event: object) -> dict[str, object]: ...
def record_runner_run_interrupted_diagnostic_event(event: object) -> None: ...
def build_runner_run_failed_diagnostic_payload(event: object) -> dict[str, object]: ...
def record_runner_run_failed_diagnostic_event(event: object) -> None: ...
def build_runner_run_completed_diagnostic_payload(event: object) -> dict[str, object]: ...
def record_runner_run_completed_diagnostic_event(event: object) -> None: ...
def build_runner_jax_runtime_configuration_started_diagnostic_payload() -> dict[str, object]: ...
def record_runner_jax_runtime_configuration_started_diagnostic_event() -> None: ...
def build_runner_execution_plan_build_started_diagnostic_payload() -> dict[str, object]: ...
def record_runner_execution_plan_build_started_diagnostic_event() -> None: ...
def build_runner_execution_plan_prepared_diagnostic_payload(
    association_mode: str,
    phenotype_count: int,
    chunk_size: int,
    variant_limit: int | None,
    device: str,
) -> dict[str, object]: ...
def record_runner_execution_plan_prepared_diagnostic_event(
    association_mode: str,
    phenotype_count: int,
    chunk_size: int,
    variant_limit: int | None,
    device: str,
) -> None: ...
def build_runner_execution_plan_dispatch_started_diagnostic_payload(
    phenotype_count: int,
    association_mode: str,
) -> dict[str, object]: ...
def record_runner_execution_plan_dispatch_started_diagnostic_event(
    phenotype_count: int,
    association_mode: str,
) -> None: ...
def build_runner_execution_plan_finalization_started_diagnostic_payload(
    phenotype_count: int,
    association_mode: str,
) -> dict[str, object]: ...
def record_runner_execution_plan_finalization_started_diagnostic_event(
    phenotype_count: int,
    association_mode: str,
) -> None: ...
def build_runner_multi_phenotype_dispatch_started_diagnostic_payload(
    phenotype_count: int,
    association_mode: str,
) -> dict[str, object]: ...
def record_runner_multi_phenotype_dispatch_started_diagnostic_event(
    phenotype_count: int,
    association_mode: str,
) -> None: ...
def build_runner_single_phenotype_dispatch_started_diagnostic_payload(
    association_mode: str,
    phenotype: str,
) -> dict[str, object]: ...
def record_runner_single_phenotype_dispatch_started_diagnostic_event(
    association_mode: str,
    phenotype: str,
) -> None: ...
def build_runner_binary_engine_dispatch_started_diagnostic_payload(
    phenotype: str,
) -> dict[str, object]: ...
def record_runner_binary_engine_dispatch_started_diagnostic_event(
    phenotype: str,
) -> None: ...
def build_runner_linear_engine_dispatch_started_diagnostic_payload(
    phenotype: str,
) -> dict[str, object]: ...
def record_runner_linear_engine_dispatch_started_diagnostic_event(
    phenotype: str,
) -> None: ...
def build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload(
    phenotype_count: int,
) -> dict[str, object]: ...
def record_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_event(
    phenotype_count: int,
) -> None: ...
def build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload(
    phenotype_count: int,
) -> dict[str, object]: ...
def record_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_event(
    phenotype_count: int,
) -> None: ...
def build_native_cli_stdout_diagnostic_payload(
    output_text: str,
    max_payload_chars: int,
) -> dict[str, object]: ...
def record_native_cli_stdout_diagnostic_event(
    output_text: str,
    max_payload_chars: int,
) -> None: ...
def build_native_cli_stderr_diagnostic_payload(
    output_text: str,
    max_payload_chars: int,
) -> dict[str, object]: ...
def record_native_cli_stderr_diagnostic_event(
    output_text: str,
    max_payload_chars: int,
) -> None: ...
def build_native_cli_interrupted_line_diagnostic_payload(
    line: str,
) -> dict[str, object]: ...
def record_native_cli_interrupted_line_diagnostic_event(
    line: str,
) -> None: ...
def build_native_cli_failed_line_diagnostic_payload(
    line: str,
) -> dict[str, object]: ...
def record_native_cli_failed_line_diagnostic_event(
    line: str,
) -> None: ...
def build_native_cli_completed_line_diagnostic_payload(
    line: str,
) -> dict[str, object]: ...
def record_native_cli_completed_line_diagnostic_event(
    line: str,
) -> None: ...
def build_execution_run_artifacts_payload(
    association_mode: str,
    phenotype_count: int,
    output_format: str,
    output_run_directories: tuple[str, ...],
    chunks_directories: tuple[str, ...],
    effective_configs: tuple[str, ...],
    phenotype_names: tuple[str, ...],
    final_output_paths: tuple[str | None, ...],
) -> dict[str, object]: ...
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
def extend_run_manifest_metadata(
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
def default_trusted_bgen_validation_cache_directory_value() -> str: ...
def plan_trusted_bgen_validation_cache_lookup(
    validation_mode: str,
    cache_path: str,
) -> NativeTrustedBgenValidationCacheLookupPlan: ...
def build_trusted_bgen_validation_cache_payload(
    fingerprint: str,
    bgen_path: str,
    sample_count: int,
    variant_count: int,
) -> dict[str, object]: ...
def write_trusted_bgen_validation_cache_payload(
    cache_path: str,
    fingerprint: str,
    bgen_path: str,
    sample_count: int,
    variant_count: int,
) -> None: ...
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
def build_jax_runtime_policy_payload(
    device: str,
    cache_directory: str | None,
    matmul_precision: str | None,
    persistent_cache: bool,
    persistent_cache_min_entry_size_bytes: int,
    persistent_cache_min_compile_time_seconds: int,
    xla_autotune_cache: bool,
    transfer_guard: bool,
) -> dict[str, object]: ...
def build_default_local_cache_directory_value(
    temporary_root: str,
    user_name: str,
    directory_name: str,
) -> str: ...
def default_local_cache_directory_value(directory_name: str) -> str: ...
def default_local_temporary_root_value() -> str: ...
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
def complete_jax_runtime_setup_validation_payload(
    requested_device: str,
    platform_name: str,
    cache_directory: str,
    matmul_precision: str,
    persistent_cache_enabled: bool,
    persistent_cache_min_entry_size_bytes: int,
    persistent_cache_min_compile_time_seconds: int,
    xla_auxiliary_cache_mode: str,
    xla_auxiliary_cache_reason: str,
    transfer_guard_enabled: bool,
    gpu_validation_status: str,
    gpu_validation_message: str | None,
) -> dict[str, object]: ...
def build_jax_runtime_setup_diagnostic_payloads(
    requested_device: str,
    platform_name: str,
    cache_directory: str,
    matmul_precision: str,
    persistent_cache_enabled: bool,
    persistent_cache_min_entry_size_bytes: int,
    persistent_cache_min_compile_time_seconds: int,
    xla_auxiliary_cache_mode: str,
    xla_auxiliary_cache_reason: str,
    transfer_guard_enabled: bool,
    gpu_validation_status: str,
    gpu_validation_message: str | None,
) -> tuple[dict[str, object], ...]: ...
def plan_jax_runtime_config_update_payloads(
    platform_name: str,
    cache_directory: str,
    matmul_precision: str,
    persistent_cache_enabled: bool,
    persistent_cache_min_entry_size_bytes: int,
    persistent_cache_min_compile_time_seconds: int,
    xla_auxiliary_cache_mode: str,
    transfer_guard_enabled: bool,
) -> tuple[dict[str, object], ...]: ...
def plan_jax_runtime_diagnostic_record_payload(
    diagnostic_level: str,
    has_telemetry_session: bool,
) -> dict[str, object]: ...
def plan_jax_runtime_diagnostic_record(
    diagnostic_level: str,
    has_telemetry_session: bool,
) -> NativeJaxRuntimeDiagnosticRecordPlan: ...
def record_jax_runtime_diagnostic_log_event(
    event: object,
    has_telemetry_session: bool,
) -> NativeJaxRuntimeDiagnosticRecordPlan: ...
def record_jax_runtime_diagnostic_event(
    event: object,
    telemetry_session: object | None,
) -> NativeJaxRuntimeDiagnosticRecordPlan: ...
def nvidia_driver_files_are_visible_value(
    control_device_path: str,
    uvm_device_path: str,
    driver_directory_path: str,
) -> bool: ...
def default_nvidia_driver_probe_paths_payload() -> dict[str, str]: ...
def plan_jax_runtime_setup_side_effects_payload(
    requested_device: str,
    persistent_cache_enabled: bool,
) -> dict[str, object]: ...
def plan_jax_gpu_validation_payload(
    nvidia_driver_visible: bool,
    backend_initialization_failed: bool,
    device_platforms: typing.Sequence[str],
    device_descriptions: typing.Sequence[str],
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
def plan_callback_worker_start(has_started: bool) -> NativeCallbackWorkerStartPlan: ...
def plan_result_write_handoff(has_result_work_item: bool) -> NativeResultWriteHandoffPlan: ...
def plan_result_write_item_dispatch(
    result_work_item_kind: str,
    expected_result_work_item_kind: str,
) -> NativeResultWriteItemDispatchPlan: ...
def plan_dosage_work_item_dispatch(
    dosage_work_item_kind: str,
) -> NativeDosageWorkItemDispatchPlan: ...
def plan_dosage_work_item_stage_duration(
    dosage_work_item_kind: str,
    chunk_count: int,
    elapsed_seconds: float,
) -> NativeDosageWorkItemStageDurationPlan: ...
def plan_callback_worker_stop_poll(
    remaining_timeout_seconds: float,
    has_started: bool,
    has_worker_error: bool,
    is_worker_alive: bool,
) -> NativeCallbackWorkerStopPollPlan: ...
def format_dosage_callback_worker_error_message(error_message: str) -> str: ...
def format_result_callback_worker_error_message(error_message: str) -> str: ...
def plan_null_logistic_nonconvergence(
    chromosome: str,
    convergence_flags: typing.Sequence[bool],
    scalar_convergence: bool,
    phenotype_names: typing.Sequence[str] | None,
    policy: str,
) -> NativeNullLogisticNonconvergencePlan: ...
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
def plan_callback_queue_operation_observation(
    queue_name: str,
    operation_name: str,
    elapsed_seconds: float,
    blocked: bool,
) -> NativeCallbackQueueOperationObservationPlan: ...
def plan_callback_queue_backpressure_observation(
    queue_name: str,
    operation_name: str,
    queue_depth: int,
    queue_capacity: int,
    elapsed_seconds: float,
    blocked: bool,
) -> NativeCallbackQueueBackpressureObservation: ...
def plan_callback_queue_stage_observation(
    queue_name: str,
    operation_name: str,
    elapsed_seconds: float,
    blocked: bool,
) -> NativeCallbackQueueStageObservationPlan: ...
def plan_callback_queue_stage_backpressure_observation(
    queue_name: str,
    operation_name: str,
    queue_depth: int,
    queue_capacity: int,
    elapsed_seconds: float,
    blocked: bool,
) -> NativeCallbackQueueStageBackpressureObservation: ...
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
def plan_telemetry_close(
    has_telemetry_session: bool,
    is_native_telemetry_session: bool,
) -> NativeTelemetryClosePlan: ...
def close_telemetry_session_with_event(telemetry_session: object | None) -> None: ...
def plan_telemetry_event_emission(
    telemetry_enabled: bool,
    has_native_telemetry_session: bool,
) -> NativeTelemetryEventEmissionPlan: ...
def plan_telemetry_progress_emission(
    telemetry_enabled: bool,
    has_native_telemetry_session: bool,
    should_emit_progress: bool,
) -> NativeTelemetryProgressEmissionPlan: ...
def plan_stage_timing_recorder(
    stage_timing_path_configured: bool,
    force: bool,
) -> NativeStageTimingRecorderPlan: ...
def plan_timing_file_write(
    has_stage_timing_recorder: bool,
    path_configured: bool,
) -> NativeTimingFileWritePlan: ...
def build_final_timing_outputs_write_started_diagnostic_payload(
    stage_timing_path: str | None,
    profile_summary_path: str | None,
    run_id: str | None,
) -> dict[str, object]: ...
def record_final_timing_outputs_write_started_diagnostic_event(
    stage_timing_path: str | None,
    profile_summary_path: str | None,
    run_id: str | None,
) -> None: ...
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
def plan_dosage_buffer_reuse(
    buffered_shape: typing.Sequence[int],
    expected_shape: typing.Sequence[int],
) -> NativeDosageBufferReusePlan | None: ...
def plan_variant_major_dosage_batch_handoff(
    metadata_count: int,
    genotype_matrix_by_variant_count: int,
    chunk_stats_count: int,
) -> NativeVariantMajorDosageBatchHandoffPlan: ...
def plan_dosage_work_handoff(chunk_count: int) -> NativeDosageWorkHandoffPlan: ...
def build_preflight_report_payload(
    sample_count: int,
    covariate_count: int,
    chromosome_count: int,
    trusted_no_missing_diploid: bool,
) -> dict[str, object]: ...
def validate_pipeline_resume_compatibility(
    chunks_directories: typing.Sequence[str],
    existing_manifest_json_values: typing.Sequence[str | None],
    current_header_json_values: typing.Sequence[str],
    resume_mode: str,
) -> None: ...

class NativePipelineOutputInitialization:
    @property
    def output_count(self) -> int: ...
    def committed_chunk_identifier_sets(self) -> list[list[int]]: ...
    def committed_chunk_identifiers(self, output_index: int) -> list[int]: ...

class NativePipelineOutputPreparationBatch:
    def __init__(
        self,
        run_directories: typing.Sequence[str],
        chunks_directories: typing.Sequence[str],
        existing_manifest_json_values: typing.Sequence[str | None],
        current_header_json_values: typing.Sequence[str],
        resume: bool,
        resume_mode: str,
    ) -> None: ...
    @property
    def output_count(self) -> int: ...
    @property
    def resume(self) -> bool: ...
    def validate_resume_compatibility(self) -> None: ...
    def initialize(
        self,
        runtime_compatibility_token: NativeRuntimeCompatibilityToken,
    ) -> NativePipelineOutputInitialization: ...

def initialize_pipeline_output_run_batch(
    run_directories: typing.Sequence[str],
    chunks_directories: typing.Sequence[str],
    existing_manifest_json_values: typing.Sequence[str | None],
    current_header_json_values: typing.Sequence[str],
    resume: bool,
    resume_mode: str,
    runtime_compatibility_token: NativeRuntimeCompatibilityToken,
) -> NativePipelineOutputInitialization: ...
def initialize_pipeline_output_runs(
    run_directories: typing.Sequence[str],
    chunks_directories: typing.Sequence[str],
    existing_manifest_json_values: typing.Sequence[str | None],
    current_header_json_values: typing.Sequence[str],
    resume: bool,
    resume_mode: str,
    runtime_compatibility_token: NativeRuntimeCompatibilityToken,
) -> list[list[int]]: ...
def validate_single_trait_preflight_shape_payload(
    phenotype_sample_count: int,
    covariate_dimension_count: int,
    covariate_sample_count: int,
    covariate_count: int,
) -> dict[str, object]: ...
def validate_multi_trait_preflight_shape_payload(
    phenotype_dimension_count: int,
    phenotype_trait_count: int,
    phenotype_sample_count: int,
    covariate_dimension_count: int,
    covariate_sample_count: int,
    covariate_count: int,
) -> dict[str, object]: ...
def validate_binary_phenotype_case_control_counts(case_count: int, control_count: int) -> None: ...
def validate_finite_array(label: str, all_values_finite: bool) -> None: ...
def validate_covariate_matrix_rank(covariate_rank: int, covariate_count: int) -> None: ...
def validate_binary_phenotype_coding(is_binary_coded: bool) -> None: ...
def validate_single_prediction_preflight_shape(
    chromosome: str,
    prediction_shape: typing.Sequence[int],
    sample_count: int,
) -> None: ...
def validate_multi_prediction_preflight_shape(
    chromosome: str,
    prediction_shape: typing.Sequence[int],
    trait_count: int,
    sample_count: int,
) -> None: ...
def emit_diagnostic_event(level: str, event: str, message: str, fields_json: str | None = None) -> None: ...
def emit_diagnostic_event_fields(
    level: str,
    event: str,
    message: str,
    fields: object,
) -> None: ...
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
