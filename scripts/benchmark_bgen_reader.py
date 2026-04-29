#!/usr/bin/env python3
"""Benchmark native Rust BGEN reads against the legacy Python backend."""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import time
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt

from g import types
from g.io import bgen as g_bgen


@dataclasses.dataclass(frozen=True)
class PathResult:
    """Timing and checksum output for one benchmark path."""

    durations_seconds: list[float]
    mean_seconds: float
    checksum: float


@dataclasses.dataclass(frozen=True)
class BenchmarkReport:
    """Structured benchmark report for BGEN reader paths."""

    bgen_path: str
    sample_path: str | None
    chunk_size: int
    variant_limit: int
    repeat_count: int
    native_rust_read_float32: PathResult
    legacy_probability_plus_conversion: PathResult | None
    speedup_native_vs_legacy: float | None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the BGEN benchmark."""
    argument_parser = argparse.ArgumentParser(description="Benchmark BGEN read paths.")
    argument_parser.add_argument("--bgen", type=Path, default=Path("data/1kg_chr22_full.bgen"))
    argument_parser.add_argument("--sample", type=Path, default=Path("data/1kg_chr22_full.sample"))
    argument_parser.add_argument("--chunk-size", type=int, default=1024)
    argument_parser.add_argument("--variant-limit", type=int, default=8192)
    argument_parser.add_argument("--repeat-count", type=int, default=5)
    return argument_parser


def load_legacy_open_bgen() -> typing.Any | None:
    """Load the legacy Python BGEN reader when it is installed."""
    try:
        return importlib.import_module("bgen_reader").open_bgen
    except ModuleNotFoundError:
        return None


def compute_checksum(genotype_matrix: np.ndarray) -> float:
    """Compute a deterministic finite checksum for one read matrix."""
    return float(np.nansum(genotype_matrix))


def run_native_rust_read_float32(
    bgen_reader: g_bgen.BgenReader,
    sample_index_array: npt.NDArray[np.int64],
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read through the strict native Rust hot path."""
    checksum = 0.0
    for variant_start in range(0, variant_limit, chunk_size):
        variant_stop = min(variant_limit, variant_start + chunk_size)
        genotype_matrix = bgen_reader.read_float32(sample_index_array, variant_start, variant_stop)
        checksum += compute_checksum(genotype_matrix)
    return checksum


def run_legacy_probability_plus_conversion(
    legacy_bgen_reader: typing.Any,
    sample_index_array: npt.NDArray[np.int64],
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read legacy probability tensors and convert them to dosage."""
    combination_count = int(np.asarray(getattr(legacy_bgen_reader, "ncombinations"), dtype=np.int32)[0])
    is_phased = bool(np.asarray(getattr(legacy_bgen_reader, "phased"), dtype=np.bool_)[0])

    checksum = 0.0
    for variant_start in range(0, variant_limit, chunk_size):
        variant_stop = min(variant_limit, variant_start + chunk_size)
        probability_tensor = legacy_bgen_reader.read(
            index=(sample_index_array, slice(variant_start, variant_stop)),
            dtype=np.float32,
            order="C",
        )
        genotype_matrix = g_bgen.convert_probability_tensor_to_dosage(
            probability_tensor=np.asarray(probability_tensor, dtype=np.float32, order="C"),
            combination_count=combination_count,
            is_phased=is_phased,
            dtype=np.float32,
            order=types.ArrayMemoryOrder.C_CONTIGUOUS,
        )
        checksum += compute_checksum(np.asarray(genotype_matrix, dtype=np.float32, order="C"))
    return checksum


def time_operation(operation: typing.Callable[[], float], repeat_count: int) -> PathResult:
    """Warm once and repeatedly time one benchmark operation."""
    warmup_checksum = operation()
    duration_seconds: list[float] = []
    checksum = warmup_checksum
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = operation()
        duration_seconds.append(time.perf_counter() - start_time)
    return PathResult(
        durations_seconds=duration_seconds,
        mean_seconds=sum(duration_seconds) / len(duration_seconds),
        checksum=checksum,
    )


def main() -> None:
    """Run the benchmark and print a JSON report."""
    arguments = build_argument_parser().parse_args()
    sample_index_array = np.arange(0, 0, dtype=np.int64)

    with g_bgen.open_bgen(arguments.bgen, sample_path=arguments.sample) as native_bgen_reader:
        variant_limit = min(arguments.variant_limit, native_bgen_reader.variant_count)
        sample_index_array = np.arange(native_bgen_reader.sample_count, dtype=np.int64)
        native_rust_read_float32 = time_operation(
            lambda: run_native_rust_read_float32(
                bgen_reader=native_bgen_reader,
                sample_index_array=sample_index_array,
                chunk_size=arguments.chunk_size,
                variant_limit=variant_limit,
            ),
            repeat_count=arguments.repeat_count,
        )

    legacy_open_bgen = load_legacy_open_bgen()
    legacy_probability_plus_conversion: PathResult | None = None
    speedup_native_vs_legacy: float | None = None

    if legacy_open_bgen is not None:
        with legacy_open_bgen(
            arguments.bgen,
            samples_filepath=arguments.sample,
            allow_complex=True,
            verbose=False,
        ) as legacy_bgen_reader:
            legacy_probability_plus_conversion = time_operation(
                lambda: run_legacy_probability_plus_conversion(
                    legacy_bgen_reader=legacy_bgen_reader,
                    sample_index_array=sample_index_array,
                    chunk_size=arguments.chunk_size,
                    variant_limit=variant_limit,
                ),
                repeat_count=arguments.repeat_count,
            )
        if not np.isclose(
            native_rust_read_float32.checksum,
            legacy_probability_plus_conversion.checksum,
            atol=1.0e-6,
        ):
            message = (
                "Checksum mismatch between native Rust BGEN read and the legacy probability+conversion path: "
                f"{native_rust_read_float32.checksum} vs {legacy_probability_plus_conversion.checksum}."
            )
            raise ValueError(message)
        speedup_native_vs_legacy = (
            legacy_probability_plus_conversion.mean_seconds / native_rust_read_float32.mean_seconds
        )

    report = BenchmarkReport(
        bgen_path=str(arguments.bgen),
        sample_path=str(arguments.sample) if arguments.sample is not None else None,
        chunk_size=arguments.chunk_size,
        variant_limit=variant_limit,
        repeat_count=arguments.repeat_count,
        native_rust_read_float32=native_rust_read_float32,
        legacy_probability_plus_conversion=legacy_probability_plus_conversion,
        speedup_native_vs_legacy=speedup_native_vs_legacy,
    )
    print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
