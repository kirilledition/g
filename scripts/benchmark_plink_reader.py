#!/usr/bin/env python3
"""Benchmark maintained PLINK chunk-ingestion paths and host-boundary overhead."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from bed_reader import open_bed
from bed_reader._open_bed import get_num_threads

from g.io.plink import iter_genotype_chunks, read_bed_chunk
from g.io.tabular import load_aligned_sample_data

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class BenchmarkPathResult:
    """Repeated timing and checksum output for one benchmark path."""

    durations_seconds: list[float]
    mean_seconds: float
    checksum: float


@dataclass(frozen=True)
class ReaderBenchmarkReport:
    """Structured benchmark output for the supported reader paths."""

    variant_limit: int
    chunk_size: int
    repeat_count: int
    bed_handle_read: BenchmarkPathResult
    direct_float32_read: BenchmarkPathResult
    iter_python: BenchmarkPathResult
    speedup_vs_bed_handle_read: dict[str, float]
    speedup_vs_iter_python: dict[str, float]


def validate_sample_order(
    observed_individual_identifiers: np.ndarray,
    expected_individual_identifiers: np.ndarray,
) -> None:
    """Ensure benchmark reads use the aligned sample order."""
    if not np.array_equal(observed_individual_identifiers, expected_individual_identifiers):
        message = "BED sample order does not match the aligned phenotype/covariate order."
        raise ValueError(message)


def force_ready_sum(genotype_matrix: jax.Array) -> float:
    """Synchronize a genotype matrix and return a deterministic checksum."""
    return float(np.asarray(jnp.nansum(genotype_matrix)))


def read_bed_handle_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through `bed_handle.read` and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    with open_bed(str(bed_path)) as bed_handle:
        validate_sample_order(
            observed_individual_identifiers=np.asarray(bed_handle.iid)[sample_index_array],
            expected_individual_identifiers=expected_individual_identifiers,
        )

        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            genotype_matrix = jnp.asarray(
                bed_handle.read(
                    index=np.s_[sample_index_array, variant_start:variant_stop],
                    dtype=np.float32,
                    order="C",
                )
            )
            checksum += force_ready_sum(genotype_matrix)

    return checksum


def read_direct_float32_chunk_path(
    bed_prefix: Path,
    sample_index_array: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Read chunks through the optimized `read_f32` helper and return a checksum."""
    checksum = 0.0
    bed_path = bed_prefix.with_suffix(".bed")
    with open_bed(str(bed_path)) as bed_handle:
        validate_sample_order(
            observed_individual_identifiers=np.asarray(bed_handle.iid)[sample_index_array],
            expected_individual_identifiers=expected_individual_identifiers,
        )

        num_threads = get_num_threads(getattr(bed_handle, "_num_threads", None))
        for variant_start in range(0, variant_limit, chunk_size):
            variant_stop = min(variant_limit, variant_start + chunk_size)
            genotype_matrix = read_bed_chunk(
                bed_handle=bed_handle,
                bed_path=bed_path,
                sample_index_array=sample_index_array,
                variant_start=variant_start,
                variant_stop=variant_stop,
                num_threads=num_threads,
            )
            checksum += force_ready_sum(genotype_matrix)

    return checksum


def iterate_supported_genotype_chunks(
    bed_prefix: Path,
    sample_indices: np.ndarray,
    expected_individual_identifiers: np.ndarray,
    chunk_size: int,
    variant_limit: int,
) -> float:
    """Run the maintained genotype chunk iterator and return a checksum."""
    checksum = 0.0
    for genotype_chunk in iter_genotype_chunks(
        bed_prefix=bed_prefix,
        sample_indices=sample_indices,
        expected_individual_identifiers=expected_individual_identifiers,
        chunk_size=chunk_size,
        variant_limit=variant_limit,
    ):
        checksum += float(np.asarray(genotype_chunk.genotypes).sum())
        checksum += float(np.asarray(genotype_chunk.observation_count).sum())
    return checksum


def time_operation(read_operation: Callable[[], float], repeat_count: int) -> BenchmarkPathResult:
    """Warm and repeatedly time one benchmark operation."""
    warmup_checksum = read_operation()
    duration_seconds: list[float] = []
    checksum = warmup_checksum
    for _ in range(repeat_count):
        start_time = time.perf_counter()
        checksum = read_operation()
        duration_seconds.append(time.perf_counter() - start_time)
    return BenchmarkPathResult(
        durations_seconds=duration_seconds,
        mean_seconds=sum(duration_seconds) / len(duration_seconds),
        checksum=checksum,
    )


def validate_checksums(
    bed_handle_read: BenchmarkPathResult,
    direct_float32_read: BenchmarkPathResult,
    iter_python: BenchmarkPathResult,
) -> None:
    """Ensure equivalent benchmark paths produce matching checksums."""
    raw_reader_checksum = bed_handle_read.checksum
    if not np.isclose(raw_reader_checksum, direct_float32_read.checksum, atol=1.0e-6):
        message = (
            f"Raw reader checksum mismatch for direct_float32_read: {direct_float32_read.checksum} "
            f"vs {raw_reader_checksum}."
        )
        raise ValueError(message)

    if not np.isfinite(iter_python.checksum):
        message = f"Chunk iterator checksum is not finite: {iter_python.checksum}."
        raise ValueError(message)


def main() -> None:
    """Run raw-reader and chunk-iterator benchmarks on the standard chr22 dataset."""
    bed_prefix = Path("data/1kg_chr22_full")
    aligned_sample_data = load_aligned_sample_data(
        bed_prefix=bed_prefix,
        phenotype_path=Path("data/pheno_cont.txt"),
        phenotype_name="phenotype_continuous",
        covariate_path=Path("data/covariates.txt"),
        covariate_names=("age", "sex"),
        is_binary_trait=False,
    )
    sample_index_array = np.ascontiguousarray(aligned_sample_data.sample_indices, dtype=np.intp)
    variant_limit = 4096
    chunk_size = 256
    repeat_count = 5

    bed_handle_read = time_operation(
        lambda: read_bed_handle_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )
    direct_float32_read = time_operation(
        lambda: read_direct_float32_chunk_path(
            bed_prefix=bed_prefix,
            sample_index_array=sample_index_array,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )
    iter_python = time_operation(
        lambda: iterate_supported_genotype_chunks(
            bed_prefix=bed_prefix,
            sample_indices=aligned_sample_data.sample_indices,
            expected_individual_identifiers=aligned_sample_data.individual_identifiers,
            chunk_size=chunk_size,
            variant_limit=variant_limit,
        ),
        repeat_count=repeat_count,
    )

    validate_checksums(
        bed_handle_read=bed_handle_read,
        direct_float32_read=direct_float32_read,
        iter_python=iter_python,
    )

    benchmark_report = ReaderBenchmarkReport(
        variant_limit=variant_limit,
        chunk_size=chunk_size,
        repeat_count=repeat_count,
        bed_handle_read=bed_handle_read,
        direct_float32_read=direct_float32_read,
        iter_python=iter_python,
        speedup_vs_bed_handle_read={
            "bed_handle_read": 1.0,
            "direct_float32_read": bed_handle_read.mean_seconds / direct_float32_read.mean_seconds,
            "iter_python": bed_handle_read.mean_seconds / iter_python.mean_seconds,
        },
        speedup_vs_iter_python={
            "iter_python": 1.0,
            "bed_handle_read": iter_python.mean_seconds / bed_handle_read.mean_seconds,
            "direct_float32_read": iter_python.mean_seconds / direct_float32_read.mean_seconds,
        },
    )
    print(json.dumps(asdict(benchmark_report), indent=2))


if __name__ == "__main__":
    main()
