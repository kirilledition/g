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

class VariantMetadata:
    variant_start_index: int
    variant_stop_index: int
    chromosome: list[str]
    variant_identifiers: list[str]
    position: npt.NDArray[np.int64]
    allele_one: list[str]
    allele_two: list[str]

class BgenReader:
    sample_count: int
    variant_count: int
    contains_embedded_samples: bool
    bgen_path: str

    def __init__(self, bgen_path: str, trusted_no_missing_diploid: bool = False) -> None: ...
    def sample_identifiers(self) -> list[str]: ...
    def chromosome_boundary_indices(self) -> list[int]: ...
    def prepare_sample_selection(self, sample_indices: npt.NDArray[np.int64]) -> None: ...
    def clear_prepared_sample_selection(self) -> None: ...
    def reset_profile(self) -> None: ...
    def profile_snapshot(self) -> dict[str, int]: ...
    def validate_trusted_no_missing_diploid(self) -> None: ...
    def variant_metadata_slice(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str]]: ...
    def read_dosage_f32(
        self,
        sample_indices: npt.NDArray[np.int64],
        variant_start: int,
        variant_stop: int,
    ) -> npt.NDArray[np.float32]: ...
    def read_dosage_f32_prepared(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> npt.NDArray[np.float32]: ...
    def read_dosage_f32_into(
        self,
        sample_indices: npt.NDArray[np.int64],
        variant_start: int,
        variant_stop: int,
        output_array: npt.NDArray[np.float32],
    ) -> None: ...
    def read_dosage_f32_into_prepared(
        self,
        variant_start: int,
        variant_stop: int,
        output_array: npt.NDArray[np.float32],
    ) -> None: ...
    def read_preprocessed_dosage_f32_into_prepared(
        self,
        variant_start: int,
        variant_stop: int,
        output_array: npt.NDArray[np.float32],
    ) -> ChunkStats: ...
    def close(self) -> None: ...

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
    def variant_metadata_slice(
        self,
        variant_start: int,
        variant_stop: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str]]: ...
    def run_bgen_chunks(
        self,
        sample_indices: npt.NDArray[np.int64],
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
    ) -> int: ...
    def run_bgen_dosage_chunks(
        self,
        sample_indices: npt.NDArray[np.int64],
        callback: object,
        committed_chunk_identifiers: list[int] | None = None,
        prefetch_chunks: int = 1,
    ) -> int: ...
    def run_bgen_dosage_buffered_chunks(
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
    ) -> None: ...
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
    ) -> None: ...
    def write_regenie2_chunk(
        self,
        *,
        metadata: object,
        allele_one_frequency: npt.NDArray[np.float32],
        observation_count: npt.NDArray[np.int32],
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
def scan_committed_chunk_identifiers(chunks_directory: str) -> list[int]: ...
def hello_from_bin() -> str: ...
def plan_genotype_chunks(
    variant_count: int,
    chunk_size: int,
    chromosome_boundary_indices: list[int],
    variant_limit: int | None = None,
    committed_chunk_identifiers: list[int] | None = None,
) -> list[ChunkSpec]: ...
def convert_probability_tensor_to_dosage_f32(
    probability_tensor: npt.NDArray[np.float32],
    combination_count: int,
    is_phased: bool,
) -> npt.NDArray[np.float32]: ...
def convert_probability_matrix_to_dosage_f32(
    probability_matrix: npt.NDArray[np.float32],
    combination_count: int,
    is_phased: bool,
) -> npt.NDArray[np.float32]: ...
