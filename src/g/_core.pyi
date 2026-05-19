from __future__ import annotations

import numpy as np
import numpy.typing as npt

class ChunkSpec:
    variant_start_index: int
    variant_stop_index: int

class ChunkStats:
    allele_one_frequency: npt.NDArray[np.float32]
    observation_count: npt.NDArray[np.int32]
    has_missing_values: bool
    dosage_square_sum: npt.NDArray[np.float32]
    imputed_dosage_square_sum: npt.NDArray[np.float32]
    info_score: npt.NDArray[np.float32]
    minor_allele_count: npt.NDArray[np.float32]
    zero_count: npt.NDArray[np.int32]
    nonzero_count: npt.NDArray[np.int32]
    is_sparse_candidate: npt.NDArray[np.bool_]
    is_rare_sparse_firth_candidate: npt.NDArray[np.bool_]

class VariantMetadata:
    variant_start_index: int
    variant_stop_index: int
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
    def chromosome_boundary_indices(self) -> list[int]: ...
    def reset_profile(self) -> None: ...
    def profile_snapshot(self) -> dict[str, int]: ...
    def validate_trusted_no_missing_diploid(self) -> None: ...
    def variant_metadata_slice(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str]]: ...
    def run_bgen_dosage_buffered_chunks(
        self,
        sample_indices: npt.NDArray[np.int64],
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int: ...
    def run_bgen_variant_major_dosage_buffered_chunks(
        self,
        sample_indices: npt.NDArray[np.int64],
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
    def get_chromosome_predictions(self, chromosome: str) -> npt.NDArray[np.float32]: ...

class OutputWriterSession:
    def __init__(
        self,
        run_directory: str,
        chunks_directory: str,
        association_mode: str,
        writer_thread_count: int = 1,
        writer_queue_depth: int = 1,
        finalize_parquet: bool = True,
        chunks_per_arrow_file: int = 4,
        arrow_compression: str = "zstd",
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
    def abort(self) -> None: ...

def finalize_output_run_chunks(
    run_directory: str,
    chunks_directory: str,
    association_mode: str,
) -> str: ...
def finalize_output_run_chunks_to_regenie_text(chunks_directory: str, regenie_text_path: str) -> None: ...
def configure_bgen_decode_tile_variant_count(tile_variant_count: int) -> None: ...
def configure_rayon_global_thread_pool(thread_count: int) -> None: ...
def scan_committed_chunk_identifiers(chunks_directory: str) -> list[int]: ...
def validate_strict_manifest_chunks(chunks_directory: str, manifest_json: str) -> list[int]: ...
def hello_from_bin() -> str: ...
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
