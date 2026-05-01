#!/usr/bin/env python3
"""Benchmark production BGEN float32 read paths against the legacy backend."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import json
import os
import subprocess
import sys
import time
import typing
from pathlib import Path

import numpy as np
import numpy.typing as npt

from g import types
from g.io import bgen as g_bgen


class BenchmarkPathMode(enum.StrEnum):
    """Selectable BGEN reader benchmark paths."""

    READ_FLOAT32 = "read_float32"
    READ_FLOAT32_PREPARED = "read_float32_prepared"
    READ_FLOAT32_INTO_PREPARED = "read_float32_into_prepared"
    LEGACY_PROBABILITY_PLUS_CONVERSION = "legacy_probability_plus_conversion"


@dataclasses.dataclass(frozen=True)
class PathResult:
    """Timing and checksum output for one benchmark path."""

    path_mode: str
    durations_seconds: list[float]
    mean_seconds: float
    checksum: float


@dataclasses.dataclass(frozen=True)
class BenchmarkCaseReport:
    """One fully specified benchmark case."""

    bgen_path: str
    sample_path: str | None
    chunk_size: int
    variant_limit: int
    repeat_count: int
    decode_tile_variant_count: int | None
    rayon_thread_count: int | None
    trusted_no_missing_diploid: bool
    path_results: list[PathResult]
    checksum_reference_path: str


@dataclasses.dataclass(frozen=True)
class BenchmarkSweepReport:
    """Collection of benchmark cases over chunking and threading knobs."""

    cases: list[BenchmarkCaseReport]


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the BGEN benchmark."""
    argument_parser = argparse.ArgumentParser(description="Benchmark BGEN float32 reader paths.")
    argument_parser.add_argument("--bgen", type=Path, default=Path("data/1kg_chr22_full.bgen"))
    argument_parser.add_argument("--sample", type=Path, default=Path("data/1kg_chr22_full.sample"))
    argument_parser.add_argument("--chunk-size", type=int, default=8192)
    argument_parser.add_argument("--chunk-sizes", default="8192")
    argument_parser.add_argument("--variant-limit", type=int, default=16384)
    argument_parser.add_argument("--repeat-count", type=int, default=5)
    argument_parser.add_argument(
        "--path-modes",
        default="read_float32,read_float32_prepared,read_float32_into_prepared,legacy_probability_plus_conversion",
    )
    argument_parser.add_argument("--decode-tile-variant-count", type=int)
    argument_parser.add_argument("--decode-tile-variant-counts", default="")
    argument_parser.add_argument("--rayon-thread-count", type=int)
    argument_parser.add_argument("--rayon-thread-counts", default="")
    argument_parser.add_argument("--trusted-no-missing-diploid", action="store_true")
    argument_parser.add_argument("--trusted-no-missing-diploid-modes", default="")
    argument_parser.add_argument("--emit-case-json", action="store_true")
    return argument_parser


def parse_optional_int_list(raw_values: str) -> list[int | None]:
    """Parse a comma-separated integer list with an optional empty sentinel."""
    parsed_values: list[int | None] = []
    for raw_value in raw_values.split(","):
        stripped_value = raw_value.strip()
        if not stripped_value:
            continue
        if stripped_value.lower() in {"none", "default"}:
            parsed_values.append(None)
            continue
        parsed_values.append(int(stripped_value))
    return parsed_values


def parse_path_modes(raw_path_modes: str) -> list[BenchmarkPathMode]:
    """Parse the requested benchmark paths."""
    parsed_path_modes = [
        BenchmarkPathMode(raw_path_mode.strip()) for raw_path_mode in raw_path_modes.split(",") if raw_path_mode.strip()
    ]
    if not parsed_path_modes:
        message = "At least one benchmark path mode is required."
        raise ValueError(message)
    return parsed_path_modes


def parse_boolean_mode_list(raw_values: str) -> list[bool]:
    """Parse a comma-separated boolean list."""
    parsed_values: list[bool] = []
    for raw_value in raw_values.split(","):
        stripped_value = raw_value.strip().lower()
        if not stripped_value:
            continue
        if stripped_value in {"true", "trusted", "on", "1", "yes"}:
            parsed_values.append(True)
            continue
        if stripped_value in {"false", "safe", "off", "0", "no"}:
            parsed_values.append(False)
            continue
        message = f"Unrecognized boolean sweep value: {raw_value}."
        raise ValueError(message)
    return parsed_values


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
    """Read through the allocation-heavy strict float32 path."""
    checksum = 0.0
    for variant_start in range(0, variant_limit, chunk_size):
        variant_stop = min(variant_limit, variant_start + chunk_size)
        genotype_matrix = bgen_reader.read_float32(sample_index_array, variant_start, variant_stop)
        checksum += compute_checksum(genotype_matrix)
    return checksum


def run_native_rust_read_float32_prepared(
    bgen_reader: g_bgen.BgenReader,
    sample_index_array: npt.NDArray[np.int64],
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read through the prepared-sample allocation path."""
    bgen_reader.prepare_sample_selection(sample_index_array)
    checksum = 0.0
    try:
        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            genotype_matrix = bgen_reader.read_float32_prepared(variant_start, variant_stop)
            checksum += compute_checksum(genotype_matrix)
    finally:
        bgen_reader.clear_prepared_sample_selection()
    return checksum


def run_native_rust_read_float32_into_prepared(
    bgen_reader: g_bgen.BgenReader,
    sample_index_array: npt.NDArray[np.int64],
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read through the prepared-sample reusable-buffer path used in production."""
    bgen_reader.prepare_sample_selection(sample_index_array)
    checksum = 0.0
    output_array = np.empty(
        (sample_index_array.size, min(chunk_size, variant_limit)),
        dtype=np.float32,
        order="C",
    )
    try:
        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            selected_variant_count = variant_stop - variant_start
            genotype_matrix = bgen_reader.read_float32_into_prepared(
                output_array[:, :selected_variant_count],
                variant_start,
                variant_stop,
            )
            checksum += compute_checksum(genotype_matrix)
    finally:
        bgen_reader.clear_prepared_sample_selection()
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


def time_operation(
    operation: typing.Callable[[], float], repeat_count: int, path_mode: BenchmarkPathMode
) -> PathResult:
    """Warm once and repeatedly time one benchmark operation."""
    warmup_checksum = operation()
    duration_seconds: list[float] = []
    checksum = warmup_checksum
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = operation()
        duration_seconds.append(time.perf_counter() - start_time)
    return PathResult(
        path_mode=path_mode.value,
        durations_seconds=duration_seconds,
        mean_seconds=sum(duration_seconds) / len(duration_seconds),
        checksum=checksum,
    )


def build_case_report(arguments: argparse.Namespace) -> BenchmarkCaseReport:
    """Run one benchmark case in-process."""
    path_modes = parse_path_modes(arguments.path_modes)
    legacy_open_bgen = load_legacy_open_bgen()
    sample_index_array = np.arange(0, 0, dtype=np.int64)

    path_results: dict[BenchmarkPathMode, PathResult] = {}
    with g_bgen.open_bgen(
        arguments.bgen,
        sample_path=arguments.sample,
        trusted_no_missing_diploid=arguments.trusted_no_missing_diploid,
    ) as native_bgen_reader:
        variant_limit = min(arguments.variant_limit, native_bgen_reader.variant_count)
        if arguments.trusted_no_missing_diploid:
            native_bgen_reader.validate_trusted_no_missing_diploid()
        sample_index_array = np.arange(native_bgen_reader.sample_count, dtype=np.int64)
        for path_mode in path_modes:
            if path_mode == BenchmarkPathMode.LEGACY_PROBABILITY_PLUS_CONVERSION:
                continue
            operation: typing.Callable[[], float]
            if path_mode == BenchmarkPathMode.READ_FLOAT32:
                def operation() -> float:
                    return run_native_rust_read_float32(
                        native_bgen_reader,
                        sample_index_array,
                        arguments.chunk_size,
                        variant_limit,
                    )
            elif path_mode == BenchmarkPathMode.READ_FLOAT32_PREPARED:
                def operation() -> float:
                    return run_native_rust_read_float32_prepared(
                        native_bgen_reader,
                        sample_index_array,
                        arguments.chunk_size,
                        variant_limit,
                    )
            elif path_mode == BenchmarkPathMode.READ_FLOAT32_INTO_PREPARED:
                def operation() -> float:
                    return run_native_rust_read_float32_into_prepared(
                        native_bgen_reader,
                        sample_index_array,
                        arguments.chunk_size,
                        variant_limit,
                    )
            else:
                message = f"Unsupported benchmark path mode: {path_mode.value}."
                raise ValueError(message)
            path_results[path_mode] = time_operation(operation, arguments.repeat_count, path_mode)

    if BenchmarkPathMode.LEGACY_PROBABILITY_PLUS_CONVERSION in path_modes:
        if legacy_open_bgen is None:
            message = "Legacy benchmark path was requested, but bgen_reader is not installed."
            raise ModuleNotFoundError(message)
        with legacy_open_bgen(
            arguments.bgen,
            samples_filepath=arguments.sample,
            allow_complex=True,
            verbose=False,
        ) as legacy_bgen_reader:
            path_results[BenchmarkPathMode.LEGACY_PROBABILITY_PLUS_CONVERSION] = time_operation(
                lambda: run_legacy_probability_plus_conversion(
                    legacy_bgen_reader,
                    sample_index_array,
                    arguments.chunk_size,
                    variant_limit,
                ),
                arguments.repeat_count,
                BenchmarkPathMode.LEGACY_PROBABILITY_PLUS_CONVERSION,
            )

    ordered_path_results = [path_results[path_mode] for path_mode in path_modes if path_mode in path_results]
    checksum_reference_path = ordered_path_results[0].path_mode
    checksum_reference_value = ordered_path_results[0].checksum
    for path_result in ordered_path_results[1:]:
        if not np.isclose(checksum_reference_value, path_result.checksum, atol=1.0e-6):
            message = (
                "Checksum mismatch between benchmark paths: "
                f"{checksum_reference_path}={checksum_reference_value} vs "
                f"{path_result.path_mode}={path_result.checksum}."
            )
            raise ValueError(message)

    return BenchmarkCaseReport(
        bgen_path=str(arguments.bgen),
        sample_path=str(arguments.sample) if arguments.sample is not None else None,
        chunk_size=arguments.chunk_size,
        variant_limit=variant_limit,
        repeat_count=arguments.repeat_count,
        decode_tile_variant_count=arguments.decode_tile_variant_count,
        rayon_thread_count=arguments.rayon_thread_count,
        trusted_no_missing_diploid=bool(arguments.trusted_no_missing_diploid),
        path_results=ordered_path_results,
        checksum_reference_path=checksum_reference_path,
    )


def run_case_subprocess(
    arguments: argparse.Namespace,
    chunk_size: int,
    decode_tile_variant_count: int | None,
    rayon_thread_count: int | None,
    *,
    trusted_no_missing_diploid: bool,
) -> BenchmarkCaseReport:
    """Run one benchmark case in a fresh subprocess so env knobs take effect."""
    command_arguments = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--bgen",
        str(arguments.bgen),
        "--sample",
        str(arguments.sample),
        "--chunk-size",
        str(chunk_size),
        "--variant-limit",
        str(arguments.variant_limit),
        "--repeat-count",
        str(arguments.repeat_count),
        "--path-modes",
        arguments.path_modes,
        "--emit-case-json",
    ]
    if decode_tile_variant_count is not None:
        command_arguments.extend(["--decode-tile-variant-count", str(decode_tile_variant_count)])
    if rayon_thread_count is not None:
        command_arguments.extend(["--rayon-thread-count", str(rayon_thread_count)])
    if trusted_no_missing_diploid:
        command_arguments.append("--trusted-no-missing-diploid")

    environment = dict(os.environ)
    if decode_tile_variant_count is None:
        environment.pop("G_BGEN_DECODE_TILE_VARIANT_COUNT", None)
    else:
        environment["G_BGEN_DECODE_TILE_VARIANT_COUNT"] = str(decode_tile_variant_count)
    if rayon_thread_count is None:
        environment.pop("RAYON_NUM_THREADS", None)
    else:
        environment["RAYON_NUM_THREADS"] = str(rayon_thread_count)

    completed_process = subprocess.run(
        command_arguments,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    case_payload = json.loads(completed_process.stdout)
    path_results = [PathResult(**path_result_payload) for path_result_payload in case_payload["path_results"]]
    return BenchmarkCaseReport(
        bgen_path=str(case_payload["bgen_path"]),
        sample_path=typing.cast("str | None", case_payload["sample_path"]),
        chunk_size=int(case_payload["chunk_size"]),
        variant_limit=int(case_payload["variant_limit"]),
        repeat_count=int(case_payload["repeat_count"]),
        decode_tile_variant_count=typing.cast("int | None", case_payload["decode_tile_variant_count"]),
        rayon_thread_count=typing.cast("int | None", case_payload["rayon_thread_count"]),
        trusted_no_missing_diploid=bool(case_payload["trusted_no_missing_diploid"]),
        path_results=path_results,
        checksum_reference_path=str(case_payload["checksum_reference_path"]),
    )


def build_sweep_report(arguments: argparse.Namespace) -> BenchmarkSweepReport:
    """Run one or more benchmark cases, using subprocesses for sweeps."""
    chunk_sizes = parse_optional_int_list(arguments.chunk_sizes) or [arguments.chunk_size]
    decode_tile_variant_counts = parse_optional_int_list(arguments.decode_tile_variant_counts)
    rayon_thread_counts = parse_optional_int_list(arguments.rayon_thread_counts)
    trusted_no_missing_diploid_modes = parse_boolean_mode_list(arguments.trusted_no_missing_diploid_modes)

    if arguments.decode_tile_variant_count is not None:
        decode_tile_variant_counts = [arguments.decode_tile_variant_count]
    elif not decode_tile_variant_counts:
        decode_tile_variant_counts = [None]

    if arguments.rayon_thread_count is not None:
        rayon_thread_counts = [arguments.rayon_thread_count]
    elif not rayon_thread_counts:
        rayon_thread_counts = [None]

    if arguments.trusted_no_missing_diploid:
        trusted_no_missing_diploid_modes = [True]
    elif not trusted_no_missing_diploid_modes:
        trusted_no_missing_diploid_modes = [False]

    if arguments.emit_case_json:
        return BenchmarkSweepReport(cases=[build_case_report(arguments)])

    case_reports: list[BenchmarkCaseReport] = []
    for chunk_size in typing.cast("list[int]", chunk_sizes):
        for decode_tile_variant_count in decode_tile_variant_counts:
            for rayon_thread_count in rayon_thread_counts:
                for trusted_no_missing_diploid in trusted_no_missing_diploid_modes:
                    case_reports.append(
                        run_case_subprocess(
                            arguments,
                            chunk_size,
                            decode_tile_variant_count,
                            rayon_thread_count,
                            trusted_no_missing_diploid=trusted_no_missing_diploid,
                        )
                    )
    return BenchmarkSweepReport(cases=case_reports)


def main() -> None:
    """Run the benchmark and print a JSON report."""
    arguments = build_argument_parser().parse_args()
    if arguments.emit_case_json:
        if arguments.decode_tile_variant_count is not None:
            os.environ["G_BGEN_DECODE_TILE_VARIANT_COUNT"] = str(arguments.decode_tile_variant_count)
        if arguments.rayon_thread_count is not None:
            os.environ["RAYON_NUM_THREADS"] = str(arguments.rayon_thread_count)
        print(json.dumps(dataclasses.asdict(build_case_report(arguments)), indent=2))
        return

    sweep_report = build_sweep_report(arguments)
    print(json.dumps(dataclasses.asdict(sweep_report), indent=2))


if __name__ == "__main__":
    main()
