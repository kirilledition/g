"""Shared models and parsers for native BGEN reader benchmarks."""

from __future__ import annotations

import dataclasses
import enum
import typing

import numpy as np


class BenchmarkPathMode(enum.StrEnum):
    """Selectable native BGEN delivery benchmark paths."""

    VARIANT_MAJOR_BUFFERED = "variant_major_buffered"
    VARIANT_MAJOR_PACKED8_BUFFERED = "variant_major_packed8_buffered"


class SampleSelectionMode(enum.StrEnum):
    """Sample-selection shapes for native BGEN delivery benchmarks."""

    FULL = "full"
    CONTIGUOUS_HALF = "contiguous_half"
    STRIDED_HALF = "strided_half"


@dataclasses.dataclass(frozen=True)
class PathResult:
    """Timing and checksum output for one benchmark path.

    Attributes:
        path_mode: Benchmark path mode.
        durations_seconds: Measured trial durations.
        mean_seconds: Mean measured duration.
        median_seconds: Median measured duration.
        checksum: Finite-dosage checksum.

    """

    path_mode: str
    durations_seconds: list[float]
    mean_seconds: float
    median_seconds: float
    checksum: float


@dataclasses.dataclass(frozen=True)
class BenchmarkCaseReport:
    """One fully specified benchmark case.

    Attributes:
        bgen_path: BGEN input path.
        sample_path: Optional sample input path.
        chunk_size: Native chunk size.
        variant_limit: Variant cap.
        repeat_count: Measured repeat count.
        decode_tile_variant_count: Optional decode tile size.
        rayon_thread_count: Optional Rayon thread count.
        trusted_no_missing_diploid: Whether trusted decoding was enabled.
        sample_selection_mode: Sample-selection mode.
        selected_sample_count: Number of selected samples.
        path_results: Results by benchmark path.
        checksum_reference_path: Path used as the checksum reference.

    """

    bgen_path: str
    sample_path: str | None
    chunk_size: int
    variant_limit: int
    repeat_count: int
    decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    trusted_no_missing_diploid: bool
    sample_selection_mode: str
    selected_sample_count: int
    path_results: list[PathResult]
    checksum_reference_path: str


@dataclasses.dataclass(frozen=True)
class BenchmarkSweepReport:
    """Collection of benchmark cases over chunking and threading knobs.

    Attributes:
        cases: Benchmark cases.

    """

    cases: list[BenchmarkCaseReport]


def parse_path_modes(raw_path_modes: str) -> list[BenchmarkPathMode]:
    """Parse requested native benchmark paths.

    Args:
        raw_path_modes: Comma-separated path mode values.

    Returns:
        Parsed path modes.

    Raises:
        ValueError: If no path modes are present or a value is invalid.

    """
    parsed_path_modes = [
        BenchmarkPathMode(raw_path_mode.strip()) for raw_path_mode in raw_path_modes.split(",") if raw_path_mode.strip()
    ]
    if not parsed_path_modes:
        message = "At least one benchmark path mode is required."
        raise ValueError(message)
    return parsed_path_modes


def parse_sample_selection_modes(raw_sample_selection_modes: str) -> list[SampleSelectionMode]:
    """Parse requested sample-selection benchmark shapes.

    Args:
        raw_sample_selection_modes: Comma-separated sample-selection modes.

    Returns:
        Parsed sample-selection modes.

    Raises:
        ValueError: If no sample-selection modes are present or a value is invalid.

    """
    parsed_sample_selection_modes = [
        SampleSelectionMode(raw_sample_selection_mode.strip())
        for raw_sample_selection_mode in raw_sample_selection_modes.split(",")
        if raw_sample_selection_mode.strip()
    ]
    if not parsed_sample_selection_modes:
        message = "At least one sample-selection mode is required."
        raise ValueError(message)
    return parsed_sample_selection_modes


def build_sample_indices(sample_count: int, sample_selection_mode: SampleSelectionMode) -> np.ndarray:
    """Build the selected sample index vector for one benchmark case.

    Args:
        sample_count: Native sample count.
        sample_selection_mode: Sample-selection mode.

    Returns:
        Selected sample indices.

    """
    if sample_selection_mode == SampleSelectionMode.FULL:
        selected_indices = np.arange(sample_count, dtype=np.int64)
    elif sample_selection_mode == SampleSelectionMode.CONTIGUOUS_HALF:
        selected_indices = np.arange(sample_count // 2, dtype=np.int64)
    elif sample_selection_mode == SampleSelectionMode.STRIDED_HALF:
        selected_indices = np.arange(0, sample_count, 2, dtype=np.int64)
    else:
        typing.assert_never(sample_selection_mode)
    return selected_indices


def supported_path_modes(
    path_modes: list[BenchmarkPathMode], *, trusted_no_missing_diploid: bool
) -> list[BenchmarkPathMode]:
    """Return path modes valid for the current trusted-mode case.

    Args:
        path_modes: Requested path modes.
        trusted_no_missing_diploid: Whether trusted decoding is enabled.

    Returns:
        Supported path modes.

    """
    if trusted_no_missing_diploid:
        return path_modes
    return [path_mode for path_mode in path_modes if path_mode != BenchmarkPathMode.VARIANT_MAJOR_PACKED8_BUFFERED]
