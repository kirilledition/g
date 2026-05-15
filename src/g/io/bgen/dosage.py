"""BGEN probability conversion helpers backed by the native Rust core."""

from __future__ import annotations

import importlib
import typing

import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from g import types


def load_backend_core() -> typing.Any:
    """Load the native extension module for BGEN decoding helpers."""
    try:
        return importlib.import_module("g._core")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Rust core helpers are unavailable. Ensure the extension module is built.") from error


def convert_probability_tensor_to_dosage(
    probability_tensor: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    combination_count: int,
    *,
    is_phased: bool,
    dtype: type[np.float32] | type[np.float64],
    order: types.ArrayMemoryOrder,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Convert a BGEN probability tensor into additive dosages."""
    if dtype is np.float32:
        core_module = load_backend_core()
        dosage_matrix = core_module.convert_probability_tensor_to_dosage_f32(
            np.asarray(probability_tensor, dtype=np.float32, order="C"), int(combination_count), bool(is_phased)
        )
        return np.asarray(dosage_matrix, dtype=np.float32, order=order.value)
    if combination_count == 3 and not is_phased:
        return np.asarray(
            probability_tensor[:, :, 1] + (2.0 * probability_tensor[:, :, 2]), dtype=dtype, order=order.value
        )
    if combination_count == 4 and is_phased:
        return np.asarray(probability_tensor[:, :, 1] + probability_tensor[:, :, 3], dtype=dtype, order=order.value)
    raise ValueError(
        "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported."
    )


def convert_probability_matrix_to_dosage(
    probability_matrix: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    combination_count: int,
    *,
    is_phased: bool,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Convert one variant's probability matrix into additive dosages."""
    if probability_matrix.dtype == np.float32:
        core_module = load_backend_core()
        dosage_vector = core_module.convert_probability_matrix_to_dosage_f32(
            np.asarray(probability_matrix, dtype=np.float32, order="C"),
            int(combination_count),
            bool(is_phased),
        )
        return np.asarray(dosage_vector, dtype=np.float32, order="C")
    if combination_count == 3 and not is_phased:
        return probability_matrix[:, 1] + (2.0 * probability_matrix[:, 2])
    if combination_count == 4 and is_phased:
        return probability_matrix[:, 1] + probability_matrix[:, 3]
    raise ValueError(
        "Unsupported BGEN probability layout. Only diploid biallelic phased or unphased variants are supported."
    )
