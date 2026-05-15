from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ContiguousVariantSlice:
    variant_start: int
    variant_stop: int


@dataclass(frozen=True)
class ReadSelection:
    sample_index_array: npt.NDArray[np.int64]
    variant_index_array: npt.NDArray[np.int64]


@dataclass(frozen=True)
class VariantReadRun:
    variant_start: int
    variant_stop: int
    output_start: int
    output_stop: int


def normalize_axis_index(axis_index: int, axis_size: int, axis_name: str) -> int:
    normalized_axis_index = axis_index + axis_size if axis_index < 0 else axis_index
    if normalized_axis_index < 0 or normalized_axis_index >= axis_size:
        raise IndexError(f"{axis_name} index {axis_index} is out of bounds for axis size {axis_size}.")
    return normalized_axis_index


def normalize_axis_selector(axis_selector: object, axis_size: int, axis_name: str) -> npt.NDArray[np.int64]:
    if axis_selector is None:
        return np.arange(axis_size, dtype=np.int64)
    if isinstance(axis_selector, slice):
        return np.arange(*axis_selector.indices(axis_size), dtype=np.int64)
    if isinstance(axis_selector, (int, np.integer)):
        return np.asarray([normalize_axis_index(int(axis_selector), axis_size, axis_name)], dtype=np.int64)
    selector_array = np.asarray(axis_selector)
    if selector_array.dtype == np.bool_:
        if selector_array.ndim != 1 or selector_array.shape[0] != axis_size:
            raise ValueError(
                f"{axis_name} boolean selector must be one-dimensional with length {axis_size}. "
                f"Observed shape {selector_array.shape}."
            )
        return np.flatnonzero(selector_array).astype(np.int64, copy=False)
    if selector_array.ndim != 1:
        raise ValueError(f"{axis_name} selector must be one-dimensional. Observed shape {selector_array.shape}.")
    normalized_values = [
        normalize_axis_index(int(raw_axis_index), axis_size, axis_name)
        for raw_axis_index in selector_array.astype(np.int64, copy=False)
    ]
    return np.asarray(normalized_values, dtype=np.int64)


def normalize_read_selection(index: object, sample_count: int, variant_count: int) -> ReadSelection:
    if index is None:
        sample_selector = slice(None)
        variant_selector = slice(None)
    elif isinstance(index, tuple):
        if len(index) != 2:
            raise ValueError("BGEN read index tuples must contain exactly two selectors: samples and variants.")
        sample_selector, variant_selector = index
    else:
        sample_selector = index
        variant_selector = slice(None)
    return ReadSelection(
        sample_index_array=normalize_axis_selector(sample_selector, sample_count, "Sample"),
        variant_index_array=normalize_axis_selector(variant_selector, variant_count, "Variant"),
    )


def resolve_contiguous_variant_slice(variant_index_array: npt.NDArray[np.int64]) -> ContiguousVariantSlice | None:
    if variant_index_array.size == 0:
        return ContiguousVariantSlice(variant_start=0, variant_stop=0)
    if variant_index_array.size == 1:
        variant_start = int(variant_index_array[0])
        return ContiguousVariantSlice(variant_start=variant_start, variant_stop=variant_start + 1)
    if np.all(np.diff(variant_index_array) == 1):
        return ContiguousVariantSlice(variant_start=int(variant_index_array[0]), variant_stop=int(variant_index_array[-1]) + 1)
    return None


def build_variant_read_runs(variant_index_array: npt.NDArray[np.int64]) -> list[VariantReadRun]:
    if variant_index_array.size == 0:
        return []
    variant_read_runs: list[VariantReadRun] = []
    run_variant_start = int(variant_index_array[0])
    run_output_start = 0
    for output_index in range(1, int(variant_index_array.size)):
        previous_variant_index = int(variant_index_array[output_index - 1])
        current_variant_index = int(variant_index_array[output_index])
        if current_variant_index != previous_variant_index + 1:
            variant_read_runs.append(VariantReadRun(run_variant_start, previous_variant_index + 1, run_output_start, output_index))
            run_variant_start = current_variant_index
            run_output_start = output_index
    variant_read_runs.append(VariantReadRun(run_variant_start, int(variant_index_array[-1]) + 1, run_output_start, int(variant_index_array.size)))
    return variant_read_runs
