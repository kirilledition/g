from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

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
        sample_key_mode: str = "iid",
    ) -> NativeAlignedSampleData: ...
    def align_multi_sample_data(
        self,
        sample_path: str | None,
        phenotype_path: str,
        phenotype_names: list[str],
        covariate_path: str | None = None,
        covariate_names: list[str] | None = None,
        is_binary_trait: bool = False,
        sample_key_mode: str = "iid",
    ) -> NativeMultiAlignedSampleData: ...
    def align_grouped_sample_data(
        self,
        sample_path: str | None,
        phenotype_path: str,
        phenotype_names: list[str],
        covariate_path: str | None = None,
        covariate_names: list[str] | None = None,
        is_binary_trait: bool = False,
        sample_key_mode: str = "iid",
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
    ) -> int: ...
    def run_bgen_variant_major_dosage_buffered_chunks_for_native_aligned_samples(
        self,
        aligned_sample_data: NativeAlignedSampleData,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int: ...
    def run_bgen_variant_major_dosage_buffered_chunks_for_native_multi_aligned_samples(
        self,
        aligned_sample_data: NativeMultiAlignedSampleData,
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
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
        sample_key_mode: str = "iid",
    ) -> None: ...
    @staticmethod
    def from_native_aligned_sample_data(
        prediction_list_path: str,
        phenotype_name: str,
        aligned_sample_data: NativeAlignedSampleData,
        sample_key_mode: str = "iid",
    ) -> RegeniePredictionSource: ...
    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]: ...

class MultiRegeniePredictionSource:
    def __init__(
        self,
        prediction_list_path: str,
        phenotype_names: list[str],
        sample_family_identifiers: list[str],
        sample_individual_identifiers: list[str],
        sample_key_mode: str = "iid",
    ) -> None: ...
    @staticmethod
    def from_native_multi_aligned_sample_data(
        prediction_list_path: str,
        aligned_sample_data: NativeMultiAlignedSampleData,
        sample_key_mode: str = "iid",
    ) -> MultiRegeniePredictionSource: ...
    @staticmethod
    def from_native_grouped_aligned_sample_data(
        prediction_list_path: str,
        grouped_aligned_sample_data: NativeGroupedAlignedSampleData,
        sample_key_mode: str = "iid",
    ) -> list[MultiRegeniePredictionSource]: ...
    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]: ...

class NativeTelemetrySession:
    def __init__(
        self,
        stream_file: str,
        queue_size: int = 65536,
        lossy: bool = True,
        event_cap: int | None = None,
    ) -> None: ...
    def emit_json_line(self, json_line: str) -> None: ...
    def finish(self) -> None: ...

class OutputWriterSession:
    def __init__(
        self,
        run_directory: str,
        chunks_directory: str,
        association_mode: str,
        writer_thread_count: int,
        writer_queue_depth: int,
        output_format: str,
        finalize_parquet: bool,
        chunks_per_arrow_file: int,
        arrow_compression: str,
        parquet_compression: str,
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
def summarize_variant_major_dosage_chunk_stats(
    genotype_matrix_by_variant: npt.NDArray[np.float32],
) -> ChunkStats: ...
def finalize_output_run_chunks(
    run_directory: str,
    chunks_directory: str,
    association_mode: str,
    output_format: str,
) -> str: ...
def resolve_output_run_paths(output_root: str, association_mode: str, output_format: str) -> NativeOutputRunPaths: ...
def prepare_output_run(
    output_root: str,
    association_mode: str,
    output_format: str,
    resume: bool,
) -> NativePreparedOutputRun: ...
def load_run_manifest_json(run_directory: str) -> str | None: ...
def write_run_manifest_json(run_directory: str, manifest_json: str) -> None: ...
def validate_run_manifest_compatibility(manifest_json: str, current_header_json: str) -> None: ...
def read_manifest_committed_chunk_identifiers(manifest_json: str) -> list[int]: ...
def initialize_output_run(
    run_directory: str,
    chunks_directory: str,
    existing_manifest_json: str | None,
    current_header_json: str,
    resume: bool,
    resume_mode: str,
) -> NativeInitializedOutputRun: ...
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
    sample_key_mode: str = "iid",
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
    sample_key_mode: str = "iid",
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
    sample_key_mode: str = "iid",
) -> NativeGroupedAlignedSampleData: ...
def align_sample_data_from_sample_file(
    sample_path: str,
    expected_sample_count: int,
    phenotype_path: str,
    phenotype_name: str,
    covariate_path: str | None = None,
    covariate_names: list[str] | None = None,
    is_binary_trait: bool = False,
    sample_key_mode: str = "iid",
) -> NativeAlignedSampleData: ...
def align_multi_sample_data_from_sample_file(
    sample_path: str,
    expected_sample_count: int,
    phenotype_path: str,
    phenotype_names: list[str],
    covariate_path: str | None = None,
    covariate_names: list[str] | None = None,
    is_binary_trait: bool = False,
    sample_key_mode: str = "iid",
) -> NativeMultiAlignedSampleData: ...
