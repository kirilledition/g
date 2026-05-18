#!/usr/bin/env python3
"""Benchmark native REGENIE2 BGEN chunk delivery paths."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
import subprocess
import sys
import time
import typing
from pathlib import Path

import numpy as np

from g import _core


class BenchmarkPathMode(enum.StrEnum):
    """Selectable native BGEN delivery benchmark paths."""

    SAMPLE_MAJOR_BUFFERED = "sample_major_buffered"
    VARIANT_MAJOR_BUFFERED = "variant_major_buffered"


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


class ChecksumCallback:
    """Native chunk callback that accumulates finite dosage checksums."""

    def __init__(self, *, variant_major: bool) -> None:
        self.variant_major = variant_major
        self.checksum = 0.0
        self.free_buffers: list[np.ndarray] = []

    def acquire_dosage_buffer(self, sample_count: int, variant_count: int) -> np.ndarray:
        return self.acquire_buffer((sample_count, variant_count))

    def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
        return self.acquire_buffer((variant_count, sample_count))

    def acquire_buffer(self, expected_shape: tuple[int, int]) -> np.ndarray:
        if self.free_buffers:
            buffer = self.free_buffers.pop()
            if buffer.shape == expected_shape:
                return buffer
        return np.empty(expected_shape, dtype=np.float32, order="C")

    def compute_preprocessed_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        del metadata, chunk_stats
        self.checksum += float(np.nansum(genotype_matrix))
        self.free_buffers.append(genotype_matrix)

    def compute_preprocessed_variant_major_dosage_chunk(
        self,
        metadata: _core.VariantMetadata,
        genotype_matrix_by_variant: np.ndarray,
        chunk_stats: _core.ChunkStats,
    ) -> None:
        del metadata, chunk_stats
        self.checksum += float(np.nansum(genotype_matrix_by_variant))
        self.free_buffers.append(genotype_matrix_by_variant)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the native BGEN benchmark."""
    argument_parser = argparse.ArgumentParser(description="Benchmark native BGEN chunk delivery paths.")
    argument_parser.add_argument("--bgen", type=Path, default=Path("data/1kg_chr22_full.bgen"))
    argument_parser.add_argument("--sample", type=Path, default=Path("data/1kg_chr22_full.sample"))
    argument_parser.add_argument("--chunk-size", type=int, default=8192)
    argument_parser.add_argument("--chunk-sizes", default="8192")
    argument_parser.add_argument("--variant-limit", type=int, default=16384)
    argument_parser.add_argument("--repeat-count", type=int, default=5)
    argument_parser.add_argument("--path-modes", default="sample_major_buffered")
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
    """Parse the requested native benchmark paths."""
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


def run_native_delivery(arguments: argparse.Namespace, path_mode: BenchmarkPathMode, variant_limit: int) -> float:
    """Run one native delivery path and return its checksum."""
    variant_major = path_mode == BenchmarkPathMode.VARIANT_MAJOR_BUFFERED
    engine = _core.Regenie2RunEngine(
        str(arguments.bgen),
        chunk_size=arguments.chunk_size,
        variant_limit=variant_limit,
        trusted_no_missing_diploid=arguments.trusted_no_missing_diploid,
    )
    if arguments.trusted_no_missing_diploid:
        engine.validate_trusted_no_missing_diploid()
    callback = ChecksumCallback(variant_major=variant_major)
    sample_indices = np.arange(engine.sample_count, dtype=np.int64)
    if variant_major:
        engine.run_bgen_variant_major_dosage_buffered_chunks(sample_indices, callback)
    else:
        engine.run_bgen_dosage_buffered_chunks(sample_indices, callback)
    return callback.checksum


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
    variant_limit = arguments.variant_limit
    path_results = [
        time_operation(
            lambda path_mode=path_mode: run_native_delivery(arguments, path_mode, variant_limit),
            arguments.repeat_count,
            path_mode,
        )
        for path_mode in path_modes
    ]
    checksum_reference_path = path_results[0].path_mode
    checksum_reference_value = path_results[0].checksum
    for path_result in path_results[1:]:
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
        path_results=path_results,
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
    """Run one benchmark case in a fresh process with low-level env knobs."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--bgen",
        str(arguments.bgen),
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
    if arguments.sample is not None:
        command.extend(["--sample", str(arguments.sample)])
    if trusted_no_missing_diploid:
        command.append("--trusted-no-missing-diploid")
    environment = os.environ.copy()
    if decode_tile_variant_count is not None:
        environment["G_BGEN_DECODE_TILE_VARIANT_COUNT"] = str(decode_tile_variant_count)
        command.extend(["--decode-tile-variant-count", str(decode_tile_variant_count)])
    if rayon_thread_count is not None:
        environment["RAYON_NUM_THREADS"] = str(rayon_thread_count)
        command.extend(["--rayon-thread-count", str(rayon_thread_count)])
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    payload = json.loads(result.stdout)
    return BenchmarkCaseReport(
        bgen_path=payload["bgen_path"],
        sample_path=payload["sample_path"],
        chunk_size=int(payload["chunk_size"]),
        variant_limit=int(payload["variant_limit"]),
        repeat_count=int(payload["repeat_count"]),
        decode_tile_variant_count=payload["decode_tile_variant_count"],
        rayon_thread_count=payload["rayon_thread_count"],
        trusted_no_missing_diploid=bool(payload["trusted_no_missing_diploid"]),
        path_results=[PathResult(**path_result) for path_result in payload["path_results"]],
        checksum_reference_path=payload["checksum_reference_path"],
    )


def build_sweep_report(arguments: argparse.Namespace) -> BenchmarkSweepReport:
    """Run all requested native BGEN benchmark cases."""
    chunk_sizes = parse_optional_int_list(arguments.chunk_sizes) or [arguments.chunk_size]
    decode_tile_variant_counts = parse_optional_int_list(arguments.decode_tile_variant_counts) or [
        arguments.decode_tile_variant_count
    ]
    rayon_thread_counts = parse_optional_int_list(arguments.rayon_thread_counts) or [arguments.rayon_thread_count]
    trusted_modes = parse_boolean_mode_list(arguments.trusted_no_missing_diploid_modes) or [
        bool(arguments.trusted_no_missing_diploid)
    ]
    cases = [
        run_case_subprocess(
            arguments,
            chunk_size=int(chunk_size),
            decode_tile_variant_count=decode_tile_variant_count,
            rayon_thread_count=rayon_thread_count,
            trusted_no_missing_diploid=trusted_no_missing_diploid,
        )
        for chunk_size in chunk_sizes
        if chunk_size is not None
        for decode_tile_variant_count in decode_tile_variant_counts
        for rayon_thread_count in rayon_thread_counts
        for trusted_no_missing_diploid in trusted_modes
    ]
    return BenchmarkSweepReport(cases=cases)


def main() -> None:
    """Run the benchmark CLI."""
    arguments = build_argument_parser().parse_args()
    if arguments.emit_case_json:
        print(json.dumps(dataclasses.asdict(build_case_report(arguments))))
        return
    print(json.dumps(dataclasses.asdict(build_sweep_report(arguments)), indent=2))


if __name__ == "__main__":
    main()
