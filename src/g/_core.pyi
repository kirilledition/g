from __future__ import annotations

import numpy as np
import numpy.typing as npt

class PyBgenReader:
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
    def close(self) -> None: ...

def hello_from_bin() -> str: ...
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
